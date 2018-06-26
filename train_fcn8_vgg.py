#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc
from loss import loss
import numpy as np
import logging
import tensorflow as tf
import sys

import models/fcn8_vgg
import utils

batch_size=2
input_height=1024
input_width=2048

def build_input(input_list_txt="/home/pier/pier_data/tensorflow-fcn/input_list_train_cityscapes.txt"):
    def _parse_function(image_tensor):
        image = tf.read_file(image_tensor[0])
        image_sem = tf.read_file(image_tensor[1])
        image = tf.image.decode_image(image, channels = 3)
        image_sem = tf.image.decode_image( image_sem , channels = 1)
        image.set_shape([None,None,3])
        image_sem.set_shape([None,None,1])
        image = tf.cast(image,tf.float32)
        image=tf.image.resize_images(image,[image_height,image_width])
        image_sem=tf.image.resize_images(image_sem,[image_height,image_width],method=ResizeMethod.NEAREST_NEIGHBOR)
        return image , image_tensor[0], image_sem
        
    samples=[]
    samples_sem = []

    with open(input_list_txt, "r") as input_list:
        for line in input_list:
            sample,sample_sem = line.strip().split(";")
            samples.append(sample.strip())
            samples_sem.append(sample_sem.strip())

    inputs = np.stack((samples, samples_sem), axis = -1)
    
    image_tensor = tf.constant(inputs, tf.string)
    dataset = tf.data.Dataset.from_tensor_slices(image_tensor)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

with tf.Session() as sess:
    immy_a,_,immy_a_sem= build_input()
    input_images,input_sem_gt = tf.train.shuffle_batch([immy_a,immy_a_sem],batch_size,100,30,8)
    with tf.name_scope("content_vgg"):
        vgg_fcn = fcn8_vgg.FCN8VGG()
        vgg_fcn.build(input_images, debug=True, num_classes=19)
        logits=vgg_fcn.upscore32
        loss = loss(logits,input_sem_gt,19)
	optim = tf.train.AdamOptimizer(1e-4, beta1=0.9).minimize(loss)
        print('Finished building Network.')
    
    saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=2)
    
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
    sess.run(init)
    
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners()
    print('Thread running')

    print('Running the Network')
    tensors = [loss,optim,input_images,input_sem_gt]
    for step in range(300000):
        loss_value, _ , images,gt= sess.run(tensors)
        if step % 1000 ==0:
            saver.save(sess,os.path.join("checkpoint", "fcn8s"),global_step=step)
            print("Saved checkpoint")
        print("Step " , step, " loss: ",loss_value)
    coord.request_stop()
    coord.join(stop_grace_period_secs=30)


