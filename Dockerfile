FROM tensorflow/tensorflow:2.0.1-gpu-py3

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      screen \
      wget && \
    rm -rf /var/lib/apt/lists/* \
    apt-get upgrade

ENV TENSOR_HOME /home/isr
WORKDIR $TENSOR_HOME

COPY ISR ./ISR
COPY scripts ./scripts
COPY weights ./weights
COPY config.yml ./
COPY setup.py ./

RUN pip install --upgrade pip
RUN pip install -e ".[gpu]" --ignore-installed
RUN pip3 install flask 
RUN pip3 install requests
RUN pip3 install flask_cors

ENV PYTHONPATH ./ISR/:$PYTHONPATH
COPY src ./src
RUN mkdir ./src/images
ENTRYPOINT python3 ./src/server.py 