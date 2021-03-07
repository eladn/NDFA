FROM ubuntu:18.04

ARG python_version=3.8
ARG conda_cuda_version=10.2
ARG conda_cuda_shortcut=cu102
ARG pytorch_version=1.7.1
ARG torch_geometric_pytorch_version=1.7.0

MAINTAINER Elad Nachmias <eladnah@gmail.com>

ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends git curl wget ca-certificates bzip2 unzip openjdk-8-jdk-headless gnupg software-properties-common
RUN DEBIAN_FRONTEND=noninteractive apt-get install lightdm -y

RUN curl -LO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN apt-get update
RUN apt-get -y install cuda
RUN apt-get -y autoclean && apt-get -y autoremove

#RUN curl -o /root/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
#    chmod +x /root/miniconda.sh && \
#    /root/miniconda.sh -b && \
#    rm /root/miniconda.sh && \
#    /root/miniconda3/bin/conda clean -ya
#ENV PATH /root/miniconda3/bin:$PATH

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda clean -ya

# https://hub.docker.com/r/rocker/cuda/dockerfile

WORKDIR /root/ndfa
RUN conda init bash
COPY requirements.unversioned.txt .
COPY environment.unversioned.yml .
RUN sed 's/%PYTHON_VERSION%/${python_version}/g' environment.unversioned.yml > environment.yml
RUN sed 's/%CUDA_VERSION%/${conda_cuda_version}/g' environment.unversioned.yml > environment.yml
RUN sed 's/%CUDA_SHORTCUT%/${conda_cuda_shortcut}/g' environment.unversioned.yml > environment.yml
RUN sed 's/%PYTORCH_VERSION%/${pytorch_version}/g' environment.unversioned.yml > environment.yml
RUN sed 's/%TORCH_GEOMETRIC_PYTORCH_VERSION%/${torch_geometric_pytorch_version}/g' environment.unversioned.yml > environment.yml
RUN sed 's/%PYTHON_VERSION%/${python_version}/g' requirements.unversioned.txt > requirements.txt
RUN sed 's/%CUDA_VERSION%/${conda_cuda_version}/g' requirements.unversioned.txt > requirements.txt
RUN sed 's/%CUDA_SHORTCUT%/${conda_cuda_shortcut}/g' requirements.unversioned.txt > requirements.txt
RUN sed 's/%PYTORCH_VERSION%/${pytorch_version}/g' requirements.unversioned.txt > requirements.txt
RUN sed 's/%TORCH_GEOMETRIC_PYTORCH_VERSION%/${torch_geometric_pytorch_version}/g' requirements.unversioned.txt > requirements.txt
RUN conda env create -f environment.yml

#RUN ["/bin/bash", "-c", "conda install --yes --file requirements.txt"]
#RUN ["/bin/bash", "-c", "pip install -r requirements.txt"]
#ENTRYPOINT ["conda", "run", "-n", "myenv", "python3", "src/server.py"]

# Pull the environment name out of the environment.yml
RUN echo "conda activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /miniconda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

CMD ["/bin/bash"]
