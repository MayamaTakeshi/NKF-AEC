import argparse
import os
import sys

def usage():
    print("""
Usage: %(app)s src_filepath echo_filepath output_filepath output_format
Ex:    %(app)s audio.wav echo.wav aec.wav ULAW

Details:
      - output_format can be: ULAW, ALAW, PCM_16 or SAME_AS_INPUT
""" % {"app": sys.argv[0]})

def main():
    if len(sys.argv) != 5:
        usage()
        sys.exit(1)

    app, src_filepath, echo_filepath, output_filepath, output_format = sys.argv

    try:
        # Attempt a relative import first
        from . import nkf_aec_core
    except ImportError:
        # If the relative import fails, try an absolute import
        import nkf_aec_core

    model = nkf_aec_core.create_model()
    nkf_aec_core.remove_echo(model, src_filepath, echo_filepath, output_filepath, output_format)
    print("Success")

if __name__ == '__main__':
    main()
