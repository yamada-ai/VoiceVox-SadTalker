# Voicevox-SadTalker
## 概要
- 文字列，話者id, 画像id を送信すると，画像内の顔にリップシンクした動画が生成される．


## 動作環境
- windows 11
- wsl2 ubuntu 20.04 LTS
- Docker version 23.0.5, build bc4487a

## 実行例

1. git clone する

    ```sh
    $ git clone https://github.com/yamada-ai/VoiceVox-SadTalker.git
    ```

1. docker-compose で build, up する

    ```sh
    $ docker-compose build && docker-compose up
    ```

1. サーバが立ち上がるので，例として以下を実行
    
    ```sh
    $ curl -X POST  -H "Content-Type: application/json"  -d '{"text":"これはテストです", "speaker_id":1, "image_id":1}' localhost:8080/create/video/
    ```
    


https://github.com/yamada-ai/VoiceVox-SadTalker/assets/24557368/e064f5e9-bf67-4ac0-ba5a-26c95760dd14

    

## 注意事項
- videoServer/dockerfile では，nvidia/cuda:11.7.0-base-ubuntu22.04をベースにして環境構築をしている．実行するPCの環境によっては不適切になる可能性があるため，動かない場合は一度チェックして欲しい
    - 参考
        - https://hub.docker.com/r/nvidia/cuda

- メモリ10GBくらい持っていくので注意
