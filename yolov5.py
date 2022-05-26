import torch
import utils
display = utils.notebook_init()  # checks

python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images
display.Image(filename='runs/detect/exp/zidane.jpg', width=600)

# Download COCO val
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017val.zip', 'tmp.zip')
unzip -q tmp.zip -d ../datasets && rm tmp.zip

# Run YOLOv5x on COCO val
python val.py --weights yolov5x.pt --data coco.yaml --img 640 --iou 0.65 --half

# Download COCO test-dev2017
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017labels.zip', 'tmp.zip')
unzip -q tmp.zip -d ../datasets && rm tmp.zip
f="test2017.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f -d ../datasets/coco/images

# Run YOLOv5x on COCO test
!python val.py --weights yolov5x.pt --data coco.yaml --img 640 --iou 0.65 --half --task test

import wandb
wandb.login()

# Train YOLOv5s on COCO128 for 3 epochs
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache

from utils.torch_utils import profile

m1 = lambda x: x * torch.sigmoid(x)
m2 = torch.nn.SiLU()
results = profile(input=torch.randn(16, 3, 640, 640), ops=[m1, m2], n=100)

# Evolve
python train.py --img 640 --batch 64 --epochs 100 --data coco128.yaml --weights yolov5s.pt --cache --noautoanchor --evolve
d=runs/train/evolve && cp evolve.* $d && zip -r evolve.zip $d && gsutil mv evolve.zip gs://bucket  # upload results (optional)

# VOC
for b, m in zip([64, 64, 64, 32, 16], ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']):  # batch, model
  !python train.py --batch {b} --weights {m}.pt --data VOC.yaml --epochs 50 --img 512 --hyp hyp.VOC.yaml --project VOC --name {m} --cache

# TensorRT 
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip
pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com  # install
python export.py --weights yolov5s.pt --include engine --imgsz 640 640 --device 0  # export
python detect.py --weights yolov5s.engine --imgsz 640 640 --device 0  # inference
