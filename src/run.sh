#!/bin/sh

model='./models/20180402-114759.pb'
image1='../data/images/Anthony_Hopkins_0001.jpg'
image2='../data/images/Anthony_Hopkins_0002.jpg'
image3='../data/images/iuchi001.jpg'
image4='../data/images/iuchi002.jpg'
image5='../data/images/jo001.jpg'
image6='../data/images/sakanaka001.jpg'
image7='../data/images/sentaro001.jpg'

image8='../data/images/iuchi003.jpg'
image9='../data/images/sakanaka003.jpg'
image10='../data/images/sentaro003.jpg'

image11='../data/images/iuchi003_g.jpg'
image12='../data/images/sakanaka003_g.jpg'
image13='../data/images/sentaro003_g.jpg'

#2316 × 3088

#/opt/homebrew/Caskroom/miniforge/base/envs/face_mask/bin/python3 src/compare.py ${model} ${image1} ${image2} --image_size 160 --margin 32 --gpu_memory_fraction 0

#/opt/homebrew/Caskroom/miniforge/base/envs/face_mask/bin/python src/compare.py src/models/20180402-114759.pb data/images/Anthony_Hopkins_0001.jpg data/images/Anthony_Hopkins_0002.jpg --image_size 160 --margin 32 --gpu_memory_fraction 0

#python compare.py ${model} ${image3} ${image4} ${image5} ${image6} ${image7} --image_size 160 --margin 32 --gpu_memory_fraction 1

python compare.py ${model} ${image8} ${image9} ${image10} ${image11} ${image12} ${image13} --image_size 160 --margin 32 --gpu_memory_fraction 1
