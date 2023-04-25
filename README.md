# nkf-aec-tools

## Overview

This is a fork of https://github.com/fjiang9/NKF-AEC

```
@article{
 yang2022low,
 title={Low-Complexity Acoustic Echo Cancellation with Neural Kalman Filtering},
 author={Yang, Dong and Jiang, Fei and Wu, Wei and Fang, Xuefei and Cao, Muyong},
 journal={arXiv preprint arXiv:2207.11388},
 year={2022}
}
```

I have refactored the code and added some scripts to simplify its use. 

## Installation

I have prepared a package definition so that the tools can be installed this way:
```
pip3 install git+https://github.com/MayamaTakeshi/NKF-AEC@nkf_aec_tools
```

## Tools

### nkf-aec

Process a single src and echo file.

### dir-nkf-aec

Process a folder containing src and echo files (src files must end with .src.wav, and echo files must end with .ech.wav)

### nkf-aec-server

server app that listens to a Unix socket to accept requests to perform AEC.

You can start it like this:
```
$ python3 nkf_aec_server.py /tmp/nkf_aec_server0
listening on /tmp/nkf_aec_server0
```

Then you can test it by connecting to the unix socket and send a request in the format: "AEC:SRC_FILEPATH;ECHO_FILEPATH;OUTPUT_FILEPATH;OUTPUT_FORMAT\\n"

Ex:
```
$ echo "AEC:/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.src.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.ech.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/out/call.nkf_aec.wav;ULAW" | nc -U /tmp/nkf_aec_server0
ok
```

To try multiple requests (they will be queued), use socat:
```
$ for i in $(seq 1 10);do echo "AEC:/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.src.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.ech.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/out/call.nkf_aec.wav;ULAW" | socat -t 3 - unix:/tmp/nkf_aec_server0;done
ok
ok
ok
ok
ok
ok
ok
ok
ok
ok
```


