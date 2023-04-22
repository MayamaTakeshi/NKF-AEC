import argparse

def usage():
    print("""
Usage: %(app)s src_filepath echo_filepath output_filepath output_format
Ex:    %(app)s audio.wav echo.wav aec.wav ULAW

Details:
      - output_format can be: ULAW, ALAW, PCM_16 or SAME_AS_INPUT
""" % {"app": sys.argv[0]})

if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) != 5:
        usage()
        sys.exit(1)

    app, src_filepath, echo_filepath, output_filepath, output_format = sys.argv

    import nkf_aec_core # importing this is slow so we leave it to be done after all checks pass
    model = nkf_aec_core.create_model()
    nkf_aec_core.remove_echo(model, src_filepath, echo_filepath, output_filepath, output_format)
    print("Success")
