# pstage_01_image_classification
## 24조

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0
- Language : Python 3.8.5
- Ubuntu 18.04.5 LTS
- Server : V100
                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Prepare Images
```
data
  +- eval
  |  +- images
  +- train
  |  +- images
```

### Training
- `python train_24_alldata.py`
- 최고 score달성한 모델 생성 파일
- 최고 성능 달성한 parameters로 default값 지정되어 있음

- `python train_24.py`
- 모델 실험 및 검증을 위한 train/validation data split이 이루어진 학습파일

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
efficientnet-b7 | V100 | 512/2, 384/2 | 15 | 3 hours

### Inference
- `python inference_24.py`

### Datasets
- dataset_24.py
