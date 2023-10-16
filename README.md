# PICO - Python Interface for CUDA Operations

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

PICO is a Python application that exposes a REST API which wraps various pipelines from the [Diffusers](https://github.com/huggingface/diffusers) library. Diffusers is a popular Python library that integrates machine learning models for generating media such as images, audio, and video.

# Table of Contents

* [PICO](#pico---python-interface-for-cuda-operations)
* [Table of Contents](#table-of-contents)
   * [Purpose](#purpose)
   * [Usage](#usage)
      * [Requirements](#requirements)
      * [Installation](#installation)
         * [Docker Environment](#docker-environment)
         * [Python Environment](#python-environment)
      * [Runtime Arguments](#runtime-arguments)
      * [Environment Variables](#environment-variables)
      * [API Endpoints](#api-endpoints)
      * [Example Request](#example-request)
      * [Building Container Image](#building-container-image)

## Purpose

To provide any application that can communicate with a REST API the ability to invoke pipelines from the Diffusers library, such as Stable Diffusion.

## Usage

PICO can be deployed as a Docker container or directly to a Python environment.

### Requirements

  * An Nvidia GPU that is sufficient to run your workload
  * Deployed as a container:
    * Docker
    * [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
  * Deployed to a Python environment:
    * Python 3.10 (+[requirements.txt](requirements.txt))
    * [NVidia CUDA Toolkit 11](https://developer.nvidia.com/cuda-downloads)

### Installation

#### Docker Environment

Run the following Docker command to start PICO:

```
docker run -d \
    --name pico \
    --gpus all \
    --restart=always \
    -p 5088:5088 \
    -v pico-storage:/pico-data \
    -e HF_HOME=/pico-data \
    -e HF_DATASETS_CACHE=/pico-data/datasets \
    -e TRANSFORMERS_CACHE=/pico-data/hub \
    exoscape/pico:latest \
    --xformers
```

> :warning: Setting the HF_HOME, HF_DATASETS_CACHE, and TRANSFORMERS_CACHE environment variables is highly recommended in order to persist cached data in the event the container is reset.

#### Python Environment

Clone the repository and run `Main.py`:

```
python Main.py
```

### Runtime Arguments

| Name             | Description                                              | Default | Required |
| ---------------- | -------------------------------------------------------- | ------- | -------- |
| --host           | The address to listen on                                 | 0.0.0.0 | No       |
| --port, -p       | The port to listen on                                    | 5088    | No       |
| --verbose, -v    | Outputs additional information should an exception occur | N/A     | No       |
| --threadsmax, -t | The maximum number of concurrent requests to allow       | 1       | No       |
| --xformers       | Enables support for xformers to optimize diffusion       | N/A     | No       |

### Environment Variables

The following environment variables are not part of PICO itself, but can be set to customize the cache directories for Diffusers:

| Name               | Description                                                                             | Default                       |
| ------------------ | --------------------------------------------------------------------------------------- | ----------------------------- |
| HF_HOME            | The path to the directory to use for cacheing data downloaded from Hugging Face         | ~/.cache/huggingface          |
| HF_DATASETS_CACHE  | The path to the directory to use for cacheing datasets downloaded from Hugging Face     | ~/.cache/huggingface/datasets |
| TRANSFORMERS_CACHE | The path to the directory to use for cacheing transformers downloaded from Hugging Face | ~/.cache/huggingface/hub      |

Additional environment variables supported by the Diffusers library can be found [here](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

### API Endpoints

JSON serialization is expected for request and response payloads.

| Method | Path                                     | Description                                                                                                                                           | Request                                                                             | Response                                     |
| ------ | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------- |
| GET    | /api/v1/health                           | Gets the status of the API (Idle, Busy)                                                                                                               | N/A                                                                                 | { "Status": "Idle" } or { "Status": "Busy" } |
| POST   | /api/v1/stable-diffusion                 | Stable Diffusion text-to-image pipeline [(info)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img)               | [StableDiffusionRequest](API/StableDiffusionRequest.py)                             | [PicoResponse](API/PicoResponse.py)          |
| POST   | /api/v1/stable-diffusion-2               | Stable Diffusion v2 text-to-image pipeline [(info)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_2)  | [StableDiffusion2Request](API/StableDiffusion2Request.py)                           | [PicoResponse](API/PicoResponse.py)          |
| POST   | /api/v1/stable-diffusion-xl              | Stable Diffusion XL text-to-image pipeline [(info)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl) | [StableDiffusionXlRequest](API/StableDiffusionXlRequest.py)                         | [PicoResponse](API/PicoResponse.py)          |
| POST   | /api/v1/stable-diffusion-image-to-image  | Stable Diffusion image-to-image pipeline [(info)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/img2img)               | [StableDiffusionImageToImageRequest](API/StableDiffusionImageToImageRequest.py)     | [PicoResponse](API/PicoResponse.py)          |
| POST   | /api/v1/stable-diffusion-image-variation | Stable Diffusion image variation pipeline [(info)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/image_variation)      | [StableDiffusionImageVariationRequest](API/StableDiffusionImageVariationRequest.py) | [PicoResponse](API/PicoResponse.py)          |
| POST   | /api/v1/stable-diffusion-inpaint         | Stable Diffusion inpainting pipeline [(info)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/inpaint)                   | [StableDiffusionInpaintRequest](API/StableDiffusionInpaintRequest.py)               | [PicoResponse](API/PicoResponse.py)          |

### Example Request

Once PICO is up and running, it can be tested by running a `curl` command such as:

```
curl -H 'Content-Type: application/json' -d '{"Phrase":"ginger cat"}' http://127.0.0.1:5088/api/v1/stable-diffusion-xl
```

### Building Container Image

If you would like to build the container image yourself, clone the repository and run the following from within the cloned folder:

```
docker build -f Containerfile -t exoscape/pico:latest .
```