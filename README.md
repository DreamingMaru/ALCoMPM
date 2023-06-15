# Data
Multimodal Emotion Recognition in Conversation(MERC)
- Data: [KEMDy19](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR)

# Installation
## Requirements
- Python 3.8
- Pytorch 2.0.0
- Pytorch-cuda 1.13.0

## Files

<pre>
<code>
ALCoMPM
└── ALCoMPM
    ├── data-generator
    │   ├── text-audio-data-generator .py 
    │   └── text-data-generator.py 
    ├── data
    │   ├── text_dev.txt
    │   ├── text_test.txt
    │   └── text_train.txt
    ├── alcompm.py 
    ├── audio_to_vector.py 
    ├── com.py # CoMPM modules
    ├── compm.py # CoMPM modules
    └── pm.py # CoMPM modules

</code>
</pre>

### text-audio-data-generator .py
- ALCoMPM에 KEMDy19 데이터를 넣기 위해 .txt 형태로 generate하는 코드입니다. 가장 먼저 실행시켜주세요.

### text-data-generator.py
- CoMPM에 KEMDy19 데이터를 넣기 위해 .txt 형태로 generate하는 코드입니다. 위 코드를 실행한 후 실행시켜주세요.

### audio_to_vector.py
- audio 데이터를 vector로 변환하는 코드입니다. 위에 두 data-generator 코드실행 이후 이 코드를 실행시켜주세요.
  
### alcompm.py
- 저희 모델입니다. 위 세 .py 파일을 실행하여 데이터를 얻으신 후 최종적으로 이 파일을 실행시켜주시면 됩니다.
