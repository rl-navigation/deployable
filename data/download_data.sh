#! /bin/sh

echo "Downloading pretrained Places365 network (44 MB)..."
wget http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar -q -O resnet18_places365.pth.tar

echo "Downloading pretrained navigation policy (15 MB)..."
wget https://tinyurl.com/yc4ekoav -q -O checkpoint.wintermute1528108143.35.pytorch

echo "Downloading campus dataset (758 MB)..."
wget https://tinyurl.com/ya6g62rg -q -O entire-campus.pytorch

