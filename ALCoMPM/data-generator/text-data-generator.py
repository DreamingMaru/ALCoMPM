import os
import random
import pandas as pd

## Reconfigure the given data into the MELD data format.
## This is for the sake of convenience.

# set file path
path = "C:/Users/User/Desktop/KEMDy20_v1_1/"
file_list = os.listdir(path)

# The number of sessions
n = len(os.listdir(path+'annotation/'))

# write file
f_train = open(path+"text_train.txt", 'w', encoding="utf-8")
f_dev = open(path+"text_dev.txt", 'w', encoding="utf-8")
f_test = open(path+"text_test.txt", 'w', encoding="utf-8")

f_train.write("Speaker\tUtterance\tEmotion\tSentiment\n")
f_dev.write("Speaker\tUtterance\tEmotion\tSentiment\n")
f_test.write("Speaker\tUtterance\tEmotion\tSentiment\n")

# Loop for each Session csv file
for session in range(1, n+1):
    csv = pd.read_csv(path+"annotation/"+"Sess" + str(session).zfill(2) + "_eval.csv", header=1)
    
    script = "script00"
    for i in range(1, len(csv)):
        id = csv.iloc[i,3].split('_') # Segment ID like "Sess01_script01_User002M_001"
        
        # If current script ends, then move on to the next script
        if (script != id[1]): 
            script = id[1] 
            
            # train - 80% / validation - 10% / test - 10%
            # It is based on randomness, so it is not a completely accurate allocation.
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
        if Emotion == "disqust": # It seems there is a typo 'disqust' in the data
           Emotion = "disgust"

        if Emotion in ["joy", "happy"]:
            Sentiment = "positive"
        elif Emotion in ["angry", "disgust", "fear", "sadness"]:
            Sentiment = "negative"
        else:
            Sentiment = 'neutral'
        
        # Utterance is located in different directory
        # Use 'id' from csv file to obtain the corresponding utterance
        u = open(path + "wav/" + "Session" + str(session).zfill(2) + "/" + "_".join(id) + ".txt", 'r')
        Utterance = u.readline().strip()
        u.close()
        
        f.write(Speaker + "\t")
        f.write(Utterance + "\t")
        f.write(Emotion + "\t")
        f.write(Sentiment + "\n")

f_train.close()
f_dev.close()
f_test.close()
del f
