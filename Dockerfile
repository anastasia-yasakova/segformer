FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y python3 python3-numpy python python-numpy \
    pep8 python3-pip
RUN pip install numpy
RUN pip install tabulate
RUN pip install matplotlib
RUN pip install tensorboard
RUN pip install scipy
RUN pip install tqdm
RUN pip install opencv-python
