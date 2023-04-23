import argparse
import os
import socket

def usage():
    print("""
This is a server application that listens on a unix socket for AEC requests.

Usage: %(app)s unix_socket_path
Ex:    %(app)s /tmp/nkf_aec_server0

You start it like this:

$ python3 nkf_aec_server.py /tmp/nkf_aec_server0
listening on /tmp/nkf_aec_server0

Then you can test it by connecting to the unix socket and send a request in the format: "AEC:SRC_FILEPATH;ECHO_FILEPATH;OUTPUT_FILEPATH;OUTPUT_FORMAT\\n"

Ex:

$ echo "AEC:/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.src.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.ech.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/out/call.nkf_aec.wav;ULAW" | nc -U /tmp/nkf_aec_server0 
ok

To try multiple requests (they will be queued), use socat:

takeshi@takeshi-desktop:~/src/git/MayamaTakeshi/NKF-AEC/src$ for i in $(seq 1 10);do echo "AEC:/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.src.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/in_mulaw/call.ech.wav;/home/takeshi/src/git/MayamaTakeshi/NKF-AEC/src/out/call.nkf_aec.wav;ULAW" | socat -t 3 - unix:/tmp/nkf_aec_server0;done
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

""" % {"app": sys.argv[0]})

def send_response(conn, response):
    print("Response:", response)
    conn.sendall(response)

if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    app, unix_socket_path = sys.argv

    if os.path.exists(unix_socket_path):
        os.remove(unix_socket_path)

    import nkf_aec_core # importing this is slow so we leave it to be done when we are ready to start
    model = nkf_aec_core.create_model()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(unix_socket_path)

    sock.listen()
    print("listening on", unix_socket_path)

    while True:
        # Wait for a client to connect
        conn, addr = sock.accept()
        print('Connected.')

        # Receive data from the client
        data = conn.recv(4096)
        print("Received data:", data)
        if not data:
            send_response(conn, b"err:no request\n")
        elif not data.startswith(b"AEC:"):
            send_response(conn, b"ecc:invalid request\n")
        else:
            data = data.strip()
            print("Request:", data)
            tokens = data[4:].decode().split(';')
            # tokens format is: src_filepath;echo_file_path;output_filepath,output_format

            if len(tokens) != 4:
                response = b"error:invalid parameters\n"
            else:
                (src_filepath, echo_filepath, output_filepath, output_format) = tokens
                try:
                    nkf_aec_core.remove_echo(model, src_filepath, echo_filepath, output_filepath, output_format)
                    response = b"ok\n"
                except Exception as e:
                    response = bytes("error:" + str(e), 'utf-8')
            send_response(conn, response)

