FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# apt & packages

WORKDIR /workspace

RUN apt update; \
    apt upgrade -y 

RUN pip install --upgrade pip
RUN apt-get install -y

COPY . /workspace

RUN conda create -n crnn python=3.8
RUN conda install -c anaconda pillow nltk natsort
RUN conda install -c conda-forge fire python-lmdb
RUN conda install -c conda-forge opencv
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN conda install pillow=6.2.1  
RUN pip install -r requirements.txt

CMD ["python","demo.py"]