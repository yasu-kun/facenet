#!/bin/sh

model='src/models/20180402-114759.pb'
image1='data/images/Anthony_Hopkins_0001.jpg'
image2='data/images/Anthony_Hopkins_0002.jpg'
image3='data/images/iuchi_no_glass.jpg'
image4='data/images/iuchi_glass.jpg'
#2316 × 3088

#/opt/homebrew/Caskroom/miniforge/base/envs/face_mask/bin/python3 src/compare.py ${model} ${image1} ${image2} --image_size 160 --margin 32 --gpu_memory_fraction 0

#/opt/homebrew/Caskroom/miniforge/base/envs/face_mask/bin/python src/compare.py src/models/20180402-114759.pb data/images/Anthony_Hopkins_0001.jpg data/images/Anthony_Hopkins_0002.jpg --image_size 160 --margin 32 --gpu_memory_fraction 0

python src/compare.py src/models/20180402-114759.pb ${image1} ${image2} ${image3} ${image4} --image_size 160 --margin 32 --gpu_memory_fraction 1
