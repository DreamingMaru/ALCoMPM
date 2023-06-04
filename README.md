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
    │   ├── text-audio-data-generator .py # ALCoMPM에 KEMDy19 데이터를 넣기 위해 .txt 형태로 generate하는 코드입니다. 먼저 실행시켜주세요.
    │   └── text-data-generator.py # CoMPM에 KEMDy19 데이터를 넣기 위해 .txt 형태로 generate하는 코드입니다. 먼저 실행시켜주세요.
    ├── data
    │   ├── text_dev.txt
    │   ├── text_test.txt
    │   └── text_train.txt
    ├── alcompm.py # 저희 모델입니다. 최종적으로 이 파일을 실행시켜주시면 됩니다.
    ├── audio_to_vector.py # audio 데이터를 vector로 변환하는 코드입니다. data-generator 코드실행 이후 이 코드를 실행시켜주세요.
    ├── com.py # CoMPM modules
    ├── compm.py # CoMPM modules
    └── pm.py # CoMPM modules

</code>
</pre>
