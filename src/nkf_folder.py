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
import soundfile as sf
from utils import gcc_phat
import argparse
import numpy as np
import wave
import mycodecs

exp_lut = [0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
           4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
           5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
           5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

BIAS = 0x84   # define the add-in bias for 16 bit samples
CLIP = 32635

def linear2ulaw(sample):
    # Get the sample into sign-magnitude.
    sign = (sample >> 8) & 0x80      # set aside the sign
    if sign != 0:
        sample = -sample            # get magnitude
    if sample > CLIP:
        sample = CLIP               # clip the magnitude

    # Convert from 16 bit linear to ulaw.
    sample = sample + BIAS
    exponent = exp_lut[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulawbyte = ~(sign | (exponent << 4) | mantissa)

    # optional CCITT trap
    if ulawbyte == 0:
        ulawbyte = 0x02

    return ulawbyte


#v_linear2ulaw = np.vectorize(mycodecs.linear2ulaw, otypes=[np.uint8])
v_linear2ulaw = np.vectorize(linear2ulaw, otypes=[np.uint8])

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
Usage: %(app)s input_folder output_folder
Ex:    %(app)s in_folder/ out_folder/
""" % {"app": sys.argv[0]})

if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)
    app, in_folder, out_folder = sys.argv

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

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
        y, sr = sf.read(os.path.join(in_folder, src_filename))
        print("src audio", y)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        print("src tensor", y)

        align = True

        if align:
            tau = gcc_phat(y[:sr * 10], x[:sr * 10], fs=sr, interp=1)
            tau = max(0, int((tau - 0.001) * sr))
            x = torch.cat([torch.zeros(tau), x])[:y.shape[-1]]

        with torch.no_grad():
            s_hat = model(x, y)

        encode_mulaw  = True

        if encode_mulaw:
            # this produces static
            #mu_law_data = np.sign(s_hat.cpu().numpy()) * np.log1p(255 * np.abs(s_hat.cpu().numpy())) / np.log1p(255)
            #mu_law_data = (mu_law_data + 1) * 128
            #mu_law_data = np.clip(mu_law_data, 0, 255).astype(np.uint8)
            #mu_law_data = (mu_law_data * 32767).astype('int16')  # convert to int16
            #mu_law_data = mu_law_data.byteswap()

            # this one works but amplitude is increased
            #mu_law_data = np.sign(s_hat.cpu().numpy()) * np.log1p(255 * np.abs(s_hat.cpu().numpy())) / np.log1p(255)
            #mu_law_data = (mu_law_data + 1) * 128 - 128
            #mu_law_data = np.clip(mu_law_data, -128, 127).astype(np.int8)
            #mu_law_data = mu_law_data
            #mu_law_data = (mu_law_data * 32767).astype('int16')  # convert to int16
            #mu_law_data = mu_law_data.byteswap()


            # this one also works but amplitude also increases
            #s_norm = s_hat.cpu().numpy() / np.abs(s_hat.cpu().numpy()).max()
            #mu_law_data = np.sign(s_norm) * np.log1p(255 * np.abs(s_norm)) / np.log1p(255)
            #mu_law_data = (mu_law_data + 1) / 2 * 255 - 128
            #mu_law_data = np.clip(mu_law_data, -128, 127).astype(np.int8)
            #mu_law_data = (mu_law_data * 32767).astype('int16')
            #mu_law_data = mu_law_data.byteswap()

            # this produces static
            #mu_law_data = np.sign(s_hat.cpu().numpy()) * np.log1p(255 * np.abs(s_hat.cpu().numpy())) / np.log1p(255)
            #mu_law_data = (mu_law_data + 1) * 128 - 128
            #mu_law_data /= np.max(np.abs(mu_law_data))
            #mu_law_data = (mu_law_data * 32767).astype('int16')
            #mu_law_data = mu_law_data.byteswap()


            # writing to raw is almost OK
            print("s_hat", s_hat)
            print("s_hat.cpu()", s_hat.cpu())
            print("s_hat.cpu().numpy()", s_hat.cpu().numpy())
            print("s_hat.numpy().astype('int16')", s_hat.numpy().astype('int16'))
            s_hat_scaled = (s_hat * 32767).clamp(-32768, 32767)
            print("s_hat_scaled", s_hat_scaled)
            b = s_hat_scaled.cpu().numpy().astype('int16')
            print("b", b)
            mu_law_data = v_linear2ulaw(b)
            #mu_law_data[::2], mu_law_data[1::2] = mu_law_data[1::2], mu_law_data[::2].copy()

            f = open(os.path.join(out_folder, 'a.raw'), 'wb')
            f.write(mu_law_data)
            f.close()

            # this gets distortion
            #s_hat_normalized = s_hat / torch.max(torch.abs(s_hat))
            # scale the data to the range [-32768, 32767]
            #s_hat_scaled = (s_hat_normalized * 32767).clamp(-32768, 32767)
            # convert to int16
            #b = s_hat_scaled.cpu().numpy().astype('int16')
            # apply mu-law encoding
            #mu_law_data = v_linear2ulaw(b)
            # convert back to int16 and swap endianness
            #mu_law_data = mu_law_data.astype('int16').byteswap()

            #print(mu_law_data)

            # Write the audio to file as mu-law
            #with sf.SoundFile(os.path.join(out_folder, out_filename), mode='w', channels=1, samplerate=8000, subtype='ULAW', endian='LITTLE', format='WAV') as file:
            #    file.write(mu_law_data)

            #mu_law_data = s_hat.cpu().numpy()

            #mu_law_data = np.sign(mu_law_data) * np.log1p(255 * np.abs(mu_law_data)) / np.log1p(256)
            #mu_law_data = np.uint8(np.round((mu_law_data + 1) / 2 * 255))

            # convert ulaw_data to little-endian byte order
            #mu_law_data_= mu_law_data.byteswap()


            #with wave.open(os.path.join(out_folder, out_filename), 'w') as wav_file:
            #    wav_file.setnchannels(1)
           #    wav_file.setsampwidth(1)
           #     wav_file.setframerate(sr)
           #     wav_file.setcomptype('NONE', 'ULAW')
           #     wav_file.writeframes(mu_law_data)


        else: 
            import sys
            for i in s_hat.cpu().numpy():
                sys.stdout.write(str(i) + " ")
            sf.write(os.path.join(out_folder, out_filename), s_hat.cpu().numpy(), sr)

        print(src_filename + " processed successfully")
