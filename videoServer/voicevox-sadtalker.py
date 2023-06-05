import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data


from fastapi import FastAPI
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],  # 許可するフロントエンドのオリジン
    allow_origins=["*"],
    # allow_origin_regex=r"^https://example-(.+)\.com$",  # 追加で許可するフロントエンドのオリジンの正規表現
    allow_credentials=True,  # 資格情報の共有の可否
    allow_methods=["*"],  # 許可するHTTPリクエストメソッド
    allow_headers=["*"],  # フロントエンドからの認可するHTTPヘッダー情報
    expose_headers=["Example-Header"],  # フロントエンドがアクセスできるHTTPヘッダー情報
)



import uuid
import json
import requests

from dataclasses import dataclass
from typing import List
from decimal import Decimal

@dataclass(frozen=True)
class createMovie:
    text:str
    speaker_id:int
    image_id:int



def get_image_path(image_path,  iid:int):
    if iid < 0:
        return None
    images = os.listdir(image_path)
    try:
        return images[iid]
    except:
        return None

@app.post("/create/video/")
def create_video(req:createMovie):
    root_dir = "/SadTalker"

    print("create/wav")
    base_url = "http://voice-engine:50021/"
    body_q = {
        'speaker': req.speaker_id,
        "text" : req.text
    }   
    res = requests.post(base_url+"audio_query", params=body_q)
    param_v = {
        'speaker': req.speaker_id,
    } 
    res_wav = requests.post(base_url+"synthesis", params=param_v, data=json.dumps(json.loads(res.text)))
    wav_path = "/wavs/{}.wav".format(uuid.uuid4())
    with open(wav_path, mode="wb") as tmp_v:
        tmp_v.write(res_wav.content)
    
    # ----------- #
    
    print("create/video")
    image_name = get_image_path("/images/", req.image_id)
    if image_name:
        pic_path ="/images/" + image_name
    else:
        print("ERROR : the image was not found")
        return
    audio_path = wav_path
    save_dir = "/movies/"+strftime("%Y_%m_%d_%H.%M.%S")
    os.makedirs(save_dir, exist_ok=True)

    device =  "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    preprocess = "crop"
    pose_style = 0
    enhancer = None
    background_enhancer = None

    batch_size = 2
    expression_scale = 1
    input_yaw = None
    input_pitch = None
    input_roll = None

    current_code_path = sys.argv[0]
    # current_root_path = os.path.split(current_code_path)[0]
    current_root_path = "/app/SadTalker/"
    checkpoint_dir = 'checkpoints/'

    os.environ['TORCH_HOME']=current_root_path+checkpoint_dir

    path_of_lm_croper = current_root_path+checkpoint_dir+'shape_predictor_68_face_landmarks.dat'
    path_of_net_recon_model = current_root_path+checkpoint_dir+'epoch_20.pth'
    dir_of_BFM_fitting = current_root_path+checkpoint_dir+'BFM_Fitting'
    wav2lip_checkpoint = current_root_path+checkpoint_dir+ 'wav2lip.pth'

    audio2pose_checkpoint = current_root_path+checkpoint_dir+'auido2pose_00140-model.pth'
    audio2pose_yaml_path = current_root_path+'src/'+'config/'+'auido2pose.yaml'
    
    audio2exp_checkpoint = current_root_path+checkpoint_dir+'auido2exp_00300-model.pth'
    audio2exp_yaml_path = current_root_path+'src/'+'config/'+'auido2exp.yaml'

    free_view_checkpoint = current_root_path+checkpoint_dir+'facevid2vid_00189-model.pth.tar'

    # not args.preprocess == 'full'
    mapping_checkpoint = current_root_path+checkpoint_dir+'mapping_00229-model.pth.tar'
    facerender_yaml_path = current_root_path+'src/'+'config/'+'facerender.yaml'

    print("path_of_lm_croper", path_of_lm_croper, current_root_path+checkpoint_dir+'shape_predictor_68_face_landmarks.dat')

    #init model
    print(path_of_net_recon_model)
    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

    print(audio2pose_checkpoint)
    print(audio2exp_checkpoint)
    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, 
                                audio2exp_checkpoint, audio2exp_yaml_path, 
                                wav2lip_checkpoint, device)
    
    print(free_view_checkpoint)
    print(mapping_checkpoint)
    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, 
                                            facerender_yaml_path, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess, source_image_flag=True)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    ref_eyeblink_coeff_path=None
    ref_pose_coeff_path=None

    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=True)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw, input_pitch, input_roll,
                                expression_scale=expression_scale, still_mode=True, preprocess=preprocess)
    
    mp4_path = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess)
    print("path of mp4 is ", mp4_path)