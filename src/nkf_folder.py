'''
Tencent is pleased to support the open source community by making NKF-AEC available.

Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.

Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
in compliance with the License. You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
'''
import torch
import torch.nn as nn
import soundfile as sf # mu-law
from utils import gcc_phat
import argparse
import numpy as np
import wave
import mycodecs

#def setcomptype_ulaw(self, type, name):
#    if type != "ULAW":
#        raise ValueError("unsupported compression type")
#    self._comptype = type
#    self._compname = name
#
# monkeypatch
#wave.Wave_write.setcomptype = setcomptype_ulaw

v_linear2ulaw = np.vectorize(mycodecs.linear2ulaw, otypes=[np.uint8])
v_linear2alaw = np.vectorize(mycodecs.linear2alaw, otypes=[np.uint8])

class ComplexGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True, dropout=0,
                 bidirectional=False):
        super().__init__()
        self.gru_r = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.gru_i = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, h_rr=None, h_ir=None, h_ri=None, h_ii=None):
        Frr, h_rr = self.gru_r(x.real, h_rr)
        Fir, h_ir = self.gru_r(x.imag, h_ir)
        Fri, h_ri = self.gru_i(x.real, h_ri)
        Fii, h_ii = self.gru_i(x.imag, h_ii)
        y = torch.complex(Frr - Fii, Fri + Fir)
        return y, h_rr, h_ir, h_ri, h_ii


class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.linear_real = nn.Linear(in_channel, out_channel, bias=bias)
        self.linear_imag = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self, x):
        y_real = self.linear_real(x.real)
        y_imag = self.linear_imag(x.imag)
        return torch.complex(y_real, y_imag)


class ComplexPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        return torch.complex(self.prelu(x.real), self.prelu(x.imag))


class KGNet(nn.Module):
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim

        self.fc_in = nn.Sequential(
            ComplexDense(2 * self.L + 1, fc_dim, bias=True),
            ComplexPReLU()
        )

        self.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers, bidirectional=False)

        self.fc_out = nn.Sequential(
            ComplexDense(rnn_dim, fc_dim, bias=True),
            ComplexPReLU(),
            ComplexDense(fc_dim, self.L, bias=True)
        )

    def init_hidden(self, batch_size, device):
        self.h_rr = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ir = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ri = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ii = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)

    def forward(self, input_feature):
        feat = self.fc_in(input_feature).unsqueeze(1)
        rnn_out, self.h_rr, self.h_ir, self.h_ri, self.h_ii = self.complex_gru(feat, self.h_rr, self.h_ir, self.h_ri, self.h_ii)
        kg = self.fc_out(rnn_out).permute(0, 2, 1)
        return kg


class NKF(nn.Module):
    def __init__(self, L=4):
        super().__init__()
        self.L = L
        self.kg_net = KGNet(L=self.L, fc_dim=18, rnn_layers=1, rnn_dim=18)
        self.stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024),
                                         return_complex=True)
        self.istft = lambda X: torch.istft(X, n_fft=1024, hop_length=256, win_length=1024,
                                           window=torch.hann_window(1024), return_complex=False)

    def forward(self, x, y):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        x = self.stft(x)
        y = self.stft(y)
        B, F, T = x.shape
        device = x.device
        h_prior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)
        h_posterior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)
        self.kg_net.init_hidden(B * F, device)

        x = x.contiguous().view(B * F, T)
        y = y.contiguous().view(B * F, T)
        echo_hat = torch.zeros(B * F, T, dtype=torch.complex64, device=device)

        for t in range(T):
            if t < self.L:
                xt = torch.cat([torch.zeros(B * F, self.L - t - 1, dtype=torch.complex64, device=device), x[:, :t + 1]],
                               dim=-1)
            else:
                xt = x[:, t - self.L + 1:t + 1]
            if xt.abs().mean() < 1e-5:
                continue

            dh = h_posterior - h_prior
            h_prior = h_posterior
            e = y[:, t] - torch.matmul(xt.unsqueeze(1), h_prior).squeeze()

            input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)
            kg = self.kg_net(input_feature)
            h_posterior = h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))

            echo_hat[:, t] = torch.matmul(xt.unsqueeze(1), h_posterior).squeeze()

        s_hat = self.istft(y - echo_hat).squeeze()

        return s_hat

def usage():
    print("""
Usage: %(app)s input_folder output_folder output_format
Ex:    %(app)s in_folder/ out_folder/ ULAW

Details:
      - output_format can be: ULAW, ALAW, PCM_16 or SAME_AS_INPUT
""" % {"app": sys.argv[0]})

if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) != 4:
        usage()
        sys.exit(1)

    app, in_folder, out_folder, output_format = sys.argv

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if not output_format in ['ULAW', 'ALAW', 'PCM_16', 'SAME_AS_INPUT']:
        sys.stderr.write("Invalid output_format " + output_format)
        usage()
        sys.exit(1)

    model = NKF(L=4)

    numparams = 0
    for f in model.parameters():
        numparams += f.numel()
    print('Total number of parameters: {:,}'.format(numparams))
    model.load_state_dict(torch.load('./nkf_epoch70.pt'), strict=True)
    model.eval()

    for i in os.scandir(in_folder):
        if not i.is_file() or not i.name.endswith('.src.wav'):
            continue
        src_filename = i.name
        print("Processing", src_filename)

        echo_src_filename = src_filename[:-8] + ".ech.wav"
        out_filename = src_filename[:-8] + ".nkf_aec.wav"

        x, sr = sf.read(os.path.join(in_folder, echo_src_filename))
        #y, sr = sf.read(os.path.join(in_folder, src_filename))

        with sf.SoundFile(os.path.join(in_folder, src_filename), 'r') as file:
           input_format = file.subtype
           y = file.read()

        print("input_format", input_format)
        if output_format == 'SAME_AS_INPUT':
            output_format = input_format

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        align = True

        if align:
            tau = gcc_phat(y[:sr * 10], x[:sr * 10], fs=sr, interp=1)
            tau = max(0, int((tau - 0.001) * sr))
            x = torch.cat([torch.zeros(tau), x])[:y.shape[-1]]

        with torch.no_grad():
            s_hat = model(x, y)

        
        if output_format in ['ULAW', 'ALAW']:
            # need to scale data to int16 range    
            s_hat_scaled = (s_hat * 32767).clamp(-32768, 32767)
            # get as array of int16 
            b = s_hat_scaled.cpu().numpy().astype('int16')
            # convert from linear to mu-law or a-law
            if output_format == 'ULAW': 
                format_id = 7
                data = v_linear2ulaw(b)
            else:
                format_id = 6
                data = v_linear2alaw(b)

            data_= data.byteswap()
            with wave.open(os.path.join(out_folder, out_filename), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(1)
                wav_file.setframerate(sr)
                wav_file.setcomptype('NONE', 'NONE')
                wav_file.writeframes(data)
                # we need to adjust the audio format as module wave doesn't support other formats aside from PCM_16 (1)
                wav_file._file.seek(20)
                wav_file._file.write((format_id).to_bytes(2, 'little'))
        else: 
            import sys
            #for i in s_hat.cpu().numpy():
            #    sys.stdout.write(str(i) + " ")
            sf.write(os.path.join(out_folder, out_filename), s_hat.cpu().numpy(), sr)

        print(src_filename + " processed successfully")
