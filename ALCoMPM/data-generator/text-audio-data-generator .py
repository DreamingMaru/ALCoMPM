import os
import random
import pandas as pd
import json

path = "C:/Users/User/Desktop/KEMDy20_v1_1/"
file_list = os.listdir(path)

n = len(os.listdir(path+'annotation/')) # n = 40

with open(path + "total_dataset.json", 'r') as file:
    sound = json.load(file)

f_train = open(path+"text_audio_train.txt", 'w', encoding="utf-8")
f_dev = open(path+"text_audio_dev.txt", 'w', encoding="utf-8")
f_test = open(path+"text_audio_test.txt", 'w', encoding="utf-8")

f_train.write("Speaker\tUtterance\tEmotion\tw2v\n")
f_dev.write("Speaker\tUtterance\tEmotion\tw2v\n")
f_test.write("Speaker\tUtterance\tEmotion\tw2v\n")


for session in range(1, 40):
    csv = pd.read_csv(path+"annotation/"+"Sess" + str(session).zfill(2) + "_eval.csv", header=1)

    script = "script00"
    for i in range(1, len(csv)):
        seg_ID = csv.iloc[i,3]
        key_i = sound['seg_ID'].index(seg_ID)
        w2v = sound['feature_vector'][key_i][0]

        id = seg_ID.split('_')

        if (script != id[1]):
            script = id[1]

            i = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
            if i == 0:
                f = f_train
            elif i == 1:
                f = f_dev
            elif i == 2:
                f = f_test
            
            f.write('\n')

        Speaker = id[2]
        Emotion = random.choice(csv.iloc[i, 4].split(";"))
        if Emotion == "disqust":
           Emotion = "disgust"

        u = open(path + "wav/" + "Session" + str(session).zfill(2) + "/" + "_".join(id) + ".txt", 'r')
        Utterance = u.readline().strip()
        u.close()
        
        f.write(Speaker + "\t")
        f.write(Utterance + "\t")
        f.write(Emotion + "\t")
        f.write(str(w2v) + "\n")

f_train.close()
f_dev.close()
f_test.close()
del f