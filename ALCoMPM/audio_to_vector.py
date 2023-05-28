# google colab 환경에서 작업
from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/Colab Notebooks/

import os
import shutil
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm  

# 전처리에서의 편의를 위해 오디오 파일만 따로 담은 'audio' directory 만들기
path = "KEMDy20_v1_1/wav"
os.mkdir('audio')
datafile_list = os.listdir(path)

for f_name in datafile_list:
  if 'wav' in f_name:
    shutil.move(path + f_name, 'audio' + f_name)
  else:
    pass

#-----------------------------------------전처리 함수--------------------------------------------#

# 음성정보 배열로 처리하는 함수
def load_wav(wav_path):
  wav, _ = librosa.load(wav_path, sr=8000)
  return wav

# 오디오 전처리하는 함수
def preprocess_audio(csv):
  
  df = pd.read_csv(csv)
  seg_id = list(df['Segment ID'][1:])
  
  # 모아야 할 정보
  speaker = []
  audio = []

  # 오디오 담기(csv 파일에 있는 seg id 순서대로)
  for segment in tqdm(seg_id):
    wav = load_wav('audio/'+segment+'.wav')  
    audio.append(wav)
  
  # 스크립트 구분(세션번호 + 스크립트번호), 화자 이름 담기
  script_id = []

  for id in seg_id:
    id = id.split('_')
    script_id.append(id[0] + '_' + id[1])
    speaker.append(id[2])

  return seg_id, script_id, speaker, audio  # 리스트 출력, 정확히 담화 순서대로 담겨있음

#------------------------------------------전처리 함수--------------------------------------------#    

total_seg_id = []
total_script_id = []
total_speaker = []
total_audio = []


