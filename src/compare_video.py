"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.models import load_model 

import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import sqlite3
import cv2

import detect_mask

def main():

    #images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
            facenet_model = './models/20180402-114759.pb'
            # Load the model
            facenet.load_model(facenet_model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # # Run forward pass to calculate embeddings
            # feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            # emb = sess.run(embeddings, feed_dict=feed_dict)
            
            # nrof_images = len(args.image_files)

            facemask_model = load_model("./models/mask_detector.model")
            cap = cv2.VideoCapture('../data/movies/no_glass.mp4')
            #cap = cv2.VideoCapture(0)
            while True:
                #今のフレームと保存されている画像たちとの類似度行列を作る．
                #閾値より近い距離にあれば，その人の顔とする．
                #無ければunknown
                tick = cv2.getTickCount()
                _, frame = cap.read()
                #顔の画像だけを抽出
                
                # images = get_face_img(frame, 160, 32, facemask_model)
                try:
                    images_list,box_list = get_face_img(frame, 160, 32, facemask_model)
                    
                except:
                    continue
                # Run forward pass to calculate embeddings
                # feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                # emb = sess.run(embeddings, feed_dict=feed_dict)
                
                # detect_name = face_matching(emb)

                fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)

                for num,images in enumerate(images_list):
                    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    detect_name = face_matching(emb)
                    # cv2.FONT_HERSHEY_PLAIN

                    cv2.putText(frame,detect_name,
                                (int(box_list[num][0]+30), int(box_list[num][1])-30),cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    box_list = [[int(i) for i in box_list[num]]]
                    #print(box_list)
                    #print((box_list[num][0],box_list[num][2]))
                    cv2.rectangle(frame,(box_list[num][0],box_list[num][1]),(box_list[num][2],box_list[num][3]),(0,255,0),thickness=2)

                
                cv2.putText(frame, "FPS:{} ".format(int(fps)),
                            (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.putText(frame,detect_name,
                #             (10, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('cap', frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            
            # print('Images:')
            # for i in range(nrof_images):
            #     print('%1d: %s' % (i, args.image_files[i]))
            # print('')
            
            # # Print distance matrix
            # print('Distance matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
            #         print('  %1.4f  ' % dist, end='')
            #     print('')



def face_matching(emb):
    dbname = '/home/yasu/workspace/facenet/src/register.db'
    conn = sqlite3.connect(dbname)
    cur = conn.cursor()

    select_sql = 'SELECT * FROM register_people'

    for row in cur.execute(select_sql):
        name = row[1]
        data = np.array(eval(row[2]))
        dis = np.sqrt(np.sum(np.square(np.subtract(data, emb[0,:]))))
        if dis < 0.7:
            cur.close()
            conn.close()
            #init_db(name)
            return name
    cur.close()
    conn.close()
    return 'unknown'


# def get_face_img(frame, image_size, margin, model):
#     img_list = []
#     img = frame
#     img_size = np.asarray(img.shape)[0:2]
#     det = detect_mask.mask_image(img, model)
#     bb = np.zeros(4, dtype=np.int32)
#     bb[0] = np.maximum(det[0]-margin/2, 0)
#     bb[1] = np.maximum(det[1]-margin/2, 0)
#     bb[2] = np.minimum(det[2]+margin/2, img_size[1])
#     bb[3] = np.minimum(det[3]+margin/2, img_size[0])
#     cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
#     aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#     prewhitened = facenet.prewhiten(aligned)
#     img_list.append(prewhitened)
#     images = np.stack(img_list)
#     return images


def get_face_img(frame, image_size, margin, model):
    img_list = []
    img = frame
    img_size = np.asarray(img.shape)[0:2]
    box_list = detect_mask.mask_image(img, model)
    images_list = []
    for det in box_list:
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        images = np.stack(img_list)
        images_list.append(images)
        img_list.clear()
    return images_list,box_list


                
# def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

#     minsize = 20 # minimum size of face
#     threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
#     factor = 0.709 # scale factor
    
#     print('Creating networks and loading parameters')
#     with tf.Graph().as_default():
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#         with sess.as_default():
#             pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
#     tmp_image_paths=copy.copy(image_paths)
#     img_list = []
#     for image in tmp_image_paths:
#         img = misc.imread(os.path.expanduser(image), mode='RGB')
#         img_size = np.asarray(img.shape)[0:2]
#         bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
#         if len(bounding_boxes) < 1:
#           image_paths.remove(image)
#           print("can't detect face, remove ", image)
#           continue
#         det = np.squeeze(bounding_boxes[0,0:4])
#         bb = np.zeros(4, dtype=np.int32)
#         bb[0] = np.maximum(det[0]-margin/2, 0)
#         bb[1] = np.maximum(det[1]-margin/2, 0)
#         bb[2] = np.minimum(det[2]+margin/2, img_size[1])
#         bb[3] = np.minimum(det[3]+margin/2, img_size[0])
#         cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
#         aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#         prewhitened = facenet.prewhiten(aligned)
#         img_list.append(prewhitened)
#     images = np.stack(img_list)
#     return images


if __name__ == '__main__':
    main()
