FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 as pico-dependency-stage

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
	git \
	curl \
	python3.10 \
	python3.10-distutils \
	python3-opencv

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

FROM pico-dependency-stage
WORKDIR /pico
COPY ./requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT ["python3.10", "Main.py"]