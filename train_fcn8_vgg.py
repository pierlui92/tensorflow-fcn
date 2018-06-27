#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc
from loss import loss
import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import datetime
from input_manager import *
from models.fcn8_vgg import FCN8VGG
from utils import *
import argparse
import cv2
import time

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_list_train', dest='input_list_train', default='input_list_train.txt', help='path of the input pair, image\\timage_sem')
parser.add_argument('--input_list_val_test', dest='input_list_val_test', default='input_list_val_test.txt', help='path of the input pair for validation or testing, image\\timage_sem')

parser.add_argument('--steps', dest='steps', type=int, default=300000, help='# of steps')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')

parser.add_argument('--num_classes', dest='num_classes', type=int, default=19, help='# of classes')

parser.add_argument('--resize', dest='resize', action='store_true', help='resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--load_size_w', dest='load_size_w', type=int, default=2048, help='scale images to this size')
parser.add_argument('--load_size_h', dest='load_size_h', type=int, default=1024, help='scale images to this size')

parser.add_argument('--crop', dest='crop', action='store_true', help='crop input images, default no crop')
parser.set_defaults(crop=False)
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=1024, help='then crop to this size')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=1024, help='then crop to this size')

parser.add_argument('--not_summary_image', dest='summary_image', action='store_false', help='summary images')
parser.set_defaults(summary_image=True)

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")

with tf.Session() as sess:
    ### build net ###
    image,_,image_sem= build_input(args.input_list_train)
    if args.resize:
        image=tf.image.resize_images(image,[args.image_height,args.image_width])
        image_sem=tf.image.resize_images(image_sem,[args.image_height,args.image_width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    else:
        h,w,_ = cv2.imread(open(args.input_list_train).readline().split(";")[0]).shape
        image.set_shape([h,w,3])
        image_sem.set_shape([h,w,1])
    
    if args.crop:
        crop_offset_w = tf.cond(tf.equal(tf.shape(image)[1]- args.crop_size_w,0), lambda : 0, lambda : tf.random_uniform((), minval=0, maxval= tf.shape(image)[1]- args.crop_size_w, dtype=tf.int32))
        crop_offset_h = tf.cond(tf.equal(tf.shape(image)[0]- args.crop_size_h,0), lambda : 0, lambda : tf.random_uniform((), minval=0, maxval= tf.shape(image)[0]- args.crop_size_h, dtype=tf.int32))
        image = tf.image.crop_to_bounding_box(image, crop_offset_h, crop_offset_w, args.crop_size_h, args.crop_size_w)          
        image_sem = tf.image.crop_to_bounding_box(image_sem, crop_offset_h, crop_offset_w, args.crop_size_h, args.crop_size_w)          

    input_images,input_sem_gt = tf.train.shuffle_batch([image,image_sem],args.batch_size,100,30,8)
    
    with tf.name_scope("content_vgg"):
        vgg_fcn = FCN8VGG()
        #### RICORDARSI DI REINSERIRE DROPOUT!!! TRAIN=TRUE!! ####
        vgg_fcn.build(input_images, debug=False, num_classes=args.num_classes)
        logits= vgg_fcn.upscore32
        loss = loss(logits,input_sem_gt,19)
        pred = vgg_fcn.pred_up
    
    summary_image = tf.summary.merge([
        tf.summary.image("image",input_images),
        tf.summary.image("sem_gt",color_tensorflow(input_sem_gt)),
        tf.summary.image("sem_pred",color_tensorflow(tf.expand_dims(pred,axis=-1)))])
    
    summary_scalar = tf.summary.merge([
        tf.summary.scalar("cross_entropy", loss)])

    optim = tf.train.AdamOptimizer(args.lr, args.beta1).minimize(loss)

    print('Finished building Network.')
    
    writer = tf.summary.FileWriter(args.checkpoint_dir) 
    saver = tf.train.Saver(max_to_keep=2)
    saver_5000 = tf.train.Saver(max_to_keep=0)

    init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
    sess.run(init)

    start_step = load(sess,args.checkpoint_dir)
    print("Loading last checkpoint")
    if  start_step >= 0:
        print("Restored step: ", start_step)
        print(" [*] Load SUCCESS")
    else:
        start_step=0
        print(" [!] Load failed...")   

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners()
    print('Thread running')
    print('Running the Network')
    
    total_time = 0
    for step in range(start_step, args.steps):
        start_time = time.time()
        
        loss_value, _ , images,gt= sess.run([loss,optim,input_images,input_sem_gt])
        
        total_time += time.time() - start_time
        time_left = (args.steps - step - 1)*total_time/(step + 1 - start_step)

        if step%10==0:
            summary_string = sess.run(summary_scalar)
            writer.add_summary(summary_string,step)
            print("Step " , step, " loss: ",loss_value, "Time left: ", datetime.timedelta(seconds=time_left))

        if step%100==0 and args.summary_image:
            summary_string = sess.run(summary_image)
            writer.add_summary(summary_string,step)
            print("Saved image summary",step)

        if step % 5000 ==0: 
            save(sess,saver_5000,os.path.join(args.checkpoint_dir, "fcn8s"),step=step)
            print("Saved checkpoint ", step)
        elif step % 1000 ==0:
            save(sess,saver,os.path.join(args.checkpoint_dir, "fcn8s"),step=step)
            print("Saved checkpoint ", step)
        
    coord.request_stop()
    coord.join(stop_grace_period_secs=30)


