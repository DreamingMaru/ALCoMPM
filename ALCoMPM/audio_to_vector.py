import os
import shutil
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm  

from google.colab import drive
drive.mount('/content/drive')
cd /content/drive/MyDrive/Colab Notebooks/

# 'audio' directory for audio data only
path = "KEMDy20_v1_1/wav"
os.mkdir('audio')
datafile_list = os.listdir(path)

for f_name in datafile_list:
  if 'wav' in f_name:
    shutil.move(path + f_name, 'audio' + f_name)
  else:
    pass

#-----------------------------------------raw audio preprocessing function--------------------------------------------#

# conversion raw audio to array 
def load_wav(wav_path):
  wav, _ = librosa.load(wav_path, sr=8000)
  return wav

def preprocess_audio(csv):
  
  df = pd.read_csv(csv)
  seg_id = list(df['Segment ID'][1:])
 
  speaker = []
  audio = []
  
  for segment in tqdm(seg_id):
    wav = load_wav('audio/'+segment+'.wav')  
    audio.append(wav)
 
  script_id = []

  for id in seg_id:
    id = id.split('_')
    script_id.append(id[0] + '_' + id[1])
    speaker.append(id[2])

  return seg_id, script_id, speaker, audio  

#--------------------------------------------------------------------------------------------------------------------#    

# total data collection
total_seg_id = []
total_script_id = []
total_speaker = []
total_audio = []

csv_list = sorted(os.listdir('KEMDy20_v1_1/annotation/'))

for csv_path in csv_list:
  seg_id, script_id, speaker, audio = preprocess_audio('KEMDy20_v1_1/annotation/'+csv_path)
  total_seg_id += seg_id 
  total_script_id += script_id
  total_speaker += speaker 
  total_audio += audio
  
# audio vectorizing with pretrained Wav2Vec2 model 
!pip install transformers
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch

# convert audio array(list) to tensor type
audio_ds = []
for wav in tqdm(total_audio):
  w = torch.FloatTensor(wav)
  audio_ds.append(w)

# pretrained Wav2Vec2 
processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
encoder = Wav2Vec2Model.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

#-------------------------------------------------function for vector extraction----------------------------------------------------#
# wav data encoding using wav2vec2 model
def encoding(wav_arr, processor = None, encoder = None, return_hidden_state=False):
    
    assert bool(processor) == bool(encoder)
    
    inputs = processor(wav_arr,
                       sampling_rate=16000,
                       return_attention_mask=True,
                       return_tensors="pt")
    
    Inputs = inputs.to('cuda')
    Encoder = encoder.to('cuda')
    outputs = encoder(output_hidden_states=return_hidden_state, **inputs)

    return outputs

# hidden, feature vector extraction
def extract_vector(data):
  with torch.no_grad():
    encoded = encoding(data, processor = processor, encoder = encoder, return_hidden_state=True)  # encoding 함수 사용

    hidden_vec = encoded.last_hidden_state.mean(dim=1).tolist()                                                         
    feature_vec = encoded.extract_features.mean(dim=1).tolist()

    return hidden_vec, feature_vec

#-----------------------------------------------------------------------------------------------------------------------------------#  

# hidden, feature vector
total_hidden_vector = []
total_feature_vector = []

for wav in tqdm(audio_ds):
  hidden, feature = extract_vector(wav)
  total_hidden_vector.append(hidden)
  total_feature_vector.append(feature)
  
# dictionary containing all audio information
total_dataset = {'seg_ID': total_seg_id,
                 'script_id': total_script_id,
                 'speaker': total_speaker,
                 'hidden_vector': total_hidden_vector,
                 'feature_vector': total_feature_vector}

# convert dataset to json file
import json
with open('total_dataset.json', 'w') as f : 
	json.dump(total_dataset, f)









