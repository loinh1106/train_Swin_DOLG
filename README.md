<h1>HCM AI CHALLENGE 2022 - Event Retrieval from Visual Data</h1>

---
## To do task 
- [x] DOLG
- [x] Hybrid swin transformer
- [x] Mish activation 
- [x] Centralize Gradient
---

## Requirements

```
pip install -r requirements.txt
```

---
## Setup 
**If apex folder is not exist**

```
!git clone https://github.com/NVIDIA/apex
%cd apex
!python setup.py install
```
## Prepare CSV file for train, validation (or test)
```
python prepare_data_csv.py \
--data_path /content/small_dog_cat_dataset \
--output_path /content/data \
--split_test
```
## Note
```--data_path```: Đường dẫn đến data images

```--output_path```: Đường dẫn chứa các file CSV

```--split_test```: Nếu cần tách thành 3 file train.csv, val.csv và test.csv (Tỷ lệ Train-Val-Test là 7-2-1)
## Train

```
python -u -m torch.distributed.launch --nproc_per_node=1 \
          train.py \
          --model_name dolg \
          --config_name dolg_b5_step3 \
          --trainCSVPath /content/drive/MyDrive/AIC_HCM/Image_Retrieval_from_Visual_Data/data/train.csv \
          --valCSVPath /content/drive/MyDrive/AIC_HCM/Image_Retrieval_from_Visual_Data/data/val.csv \
          --loss_name cosface \
          --use_central_gradient \
          --use_mish \ 
```

## Note
```--model_name```: **'dolg'** cho DOLG và **'swin'** cho Swin Transformer

```--config_name```: tên file config tương ứng với model

```--trainCSVPath```: Đường dẫn đến file train csv

```--valCSVPath```: Đường dẫn đến file val csv

```--loss_name```: Tên của Loss.
- arcface_dynamicmargin: ArcFaceLossAdaptiveMargin Loss
- cosface : Cosface Loss
- CE_smooth_loss : CrossEntropyLossWithLabelSmoothing Loss
- smooth_CE_loss: LabelSmoothingCrossEntropy Loss
- circleloss: CircleLoss 

```--use_central_gradient```: Sử dụng *central_gradient*, nếu không dùng chỉ cần comment nó lại

```--use_mish```: Sử dụng *Mish Function* thay cho *SiLU Activation* nếu không dùng chỉ cần comment nó lại


**Load Trained Model and Continue Training**
```
python -u -m torch.distributed.launch --nproc_per_node=1 \
          train.py \
          --model_name dolg \
          --config_name dolg_b5_step3 \
          --trainCSVPath /content/drive/MyDrive/AIC_HCM/Image_Retrieval_from_Visual_Data/data/train.csv \
          --valCSVPath /content/drive/MyDrive/AIC_HCM/Image_Retrieval_from_Visual_Data/data/val.csv \
          --checkpoint './run/saved/dolg_efficientnet_b5_ns_step3_2.pth' \
          --loss_name cosface \
          --use_central_gradient \
          --use_mish \ 
```

---
## Folder Structure

```
  Main-folder/
  │
  ├── config/ 
  │   ├── config.py - configuration
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── dataset.py
  |
  ├── model/ - this folder contains any net of your project.
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging 
  │   └── submission/ -  submission file are saved here
  │
  ├── scripts/ - main function 
  │   └── pipeline.py
  │   └── OCR.py
  │   └── segment.py
  |
  ├── test/ - test functions
  │   └── run.py
  │   └── ...
  |
  ├── tools/ - open source are saved here
  │   └── detectron2 dir
  │   └── ...
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
```
