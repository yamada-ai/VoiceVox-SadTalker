FROM nvidia/cuda:11.7.0-base-ubuntu22.04

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get -y update && apt-get -y upgrade 
RUN apt-get -y install python3-pip git wget

RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 ffmpeg tavern fastapi unicorn uvicorn

WORKDIR /app

RUN git clone https://github.com/OpenTalker/SadTalker.git 

WORKDIR /app/SadTalker

RUN pip3 install -r requirements.txt

RUN chmod +x scripts/download_models.sh && scripts/download_models.sh

RUN apt-get -y update &&  \
    apt-get install -f && \ 
    apt-get install -y libgl1-mesa-dev libglib2.0-0 ffmpeg zip

RUN cd ./checkpoints/ && unzip BFM_Fitting.zip && unzip hub.zip 

COPY voicevox-sadtalker.py /app/SadTalker