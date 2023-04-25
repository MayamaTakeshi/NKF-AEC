import argparse
import os
import sys
import socket

def usage():
    print("""
This is a server application that listens on a unix socket for AEC requests.

Usage: nkf-aec-server unix_socket_path
Ex:    nkf-aec-server /tmp/nkf_aec_server0
""")

def send_response(conn, response):
    try:
        print("Response:", response)
        conn.sendall(response)
    except Exception as e:
        print("Failed while sending response", e)

def main():
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    app, unix_socket_path = sys.argv

    if os.path.exists(unix_socket_path):
        os.remove(unix_socket_path)

    try:
        # Attempt a relative import first
        from . import nkf_aec_core
    except ImportError:
        # If the relative import fails, try an absolute import
        import nkf_aec_core

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
                    response = bytes("error:" + str(e) + "\n", 'utf-8')
            send_response(conn, response)

if __name__ == "__main__":
    main()
