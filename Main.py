import argparse
from API.PicoAPI import PicoAPI
from PipelineOptions import PipelineOptions

# The following environment variables can be set to override the default cache directory:
# - HF_HOME
# - HF_DATASETS_CACHE
# - TRANSFORMERS_CACHE
if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="The address to listen on.")
  arg_parser.add_argument(
    "-p", "--port",
    type=int,
    default=5088,
    help="The port to listen on.")
  arg_parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="Outputs additional information should an exception occur.")
  arg_parser.add_argument(
    "--xformers",
    action="store_true",
    help="Enables support for xformers to optimize diffusion.")
  args = arg_parser.parse_args()
  pipeline_options = PipelineOptions(args.xformers)

  if (pipeline_options.IsXformersEnabled):
    print("PICO: xformers support enabled.")

  print("PICO: Initializing...")
  api = PicoAPI(
    pipeline_options=pipeline_options,
    host=args.host,
    port=args.port,
    is_debug=args.verbose)
  print("PICO: Shutting down...")