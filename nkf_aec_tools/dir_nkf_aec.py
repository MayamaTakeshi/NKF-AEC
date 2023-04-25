import argparse
import os
import sys

def usage():
    print("""
Usage: dir-nkf-aec input_folder output_folder output_format
Ex:    dir-nkf-aec in_folder/ out_folder/ ULAW

Details:
      - output_format can be: ULAW, ALAW, PCM_16 or SAME_AS_INPUT
""")

def main():
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

    try:
        # Attempt a relative import first
        from . import nkf_aec_core
    except ImportError:
        # If the relative import fails, try an absolute import
        import nkf_aec_core

    model = nkf_aec_core.create_model()

    for i in os.scandir(in_folder):
        if not i.is_file() or not i.name.endswith('.src.wav'):
            continue
        src_filename = i.name

        echo_filename = src_filename[:-8] + ".ech.wav"
        output_filename = src_filename[:-8] + ".nkf_aec.wav"

        src_filepath = os.path.join(in_folder, src_filename)
        echo_filepath = os.path.join(in_folder, echo_filename)
        output_filepath = os.path.join(out_folder, output_filename)

        print("Processing", src_filepath)
        nkf_aec_core.remove_echo(model, src_filepath, echo_filepath, output_filepath, output_format)
        print("Processing", src_filepath, "completed successfully")


if __name__ == "__main__":
    main()
