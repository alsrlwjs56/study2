FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /workspace

RUN apt update; \
    apt upgrade -y 

RUN apt-get install -y

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install --upgrade pip
RUN apt-get -y install libgl1-mesa-glx
COPY . /workspace

RUN pip install craft_text_detector
RUN pip install -r requirements.txt

CMD ["python","craft.py"]


