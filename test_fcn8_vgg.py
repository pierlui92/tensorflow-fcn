import os
import numpy as np
import tensorflow as tf
from input_manager import *
from models.fcn8_vgg import FCN8VGG
from utils import *
import argparse
import cv2

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_list_val_test', dest='input_list_val_test', default='input_list_val_test.txt', help='path of the input pair for validation or testing, image\\timage_sem')

parser.add_argument('--checkpoint_path', dest='checkpoint_path', default='./checkpoint', help='checkpoint folder or path')

parser.add_argument('--test_dir', dest='test_dir', default='./test/', help='test sample are saved here')

parser.add_argument('--num_classes', dest='num_classes', type=int, default=19, help='# of classes')

parser.add_argument('--resize', dest='resize', action='store_true', help='resize input images, default full_res no resize')
parser.set_defaults(resize=False)
parser.add_argument('--load_size_w', dest='load_size_w', type=int, default=2048, help='scale images to this size')
parser.add_argument('--load_size_h', dest='load_size_h', type=int, default=1024, help='scale images to this size')

args = parser.parse_args()

if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)

with tf.Session() as sess:
    image,image_path,image_sem= build_input(args.input_list_val_test)

    if args.resize:
        image=tf.image.resize_images(image,[args.image_height,args.image_width])
        image_sem=tf.image.resize_images(image_sem,[args.image_height,args.image_width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    with tf.name_scope("content_vgg"):
        vgg_fcn = FCN8VGG()
        vgg_fcn.build(tf.expand_dims(image,axis=0), debug=False, num_classes=args.num_classes)
        pred = vgg_fcn.pred_up

    saver = tf.train.Saver()

    init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
    sess.run(init)

    start_step = load(sess,args.checkpoint_path)
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

    num_sample = len(open(args.input_list_val_test).readlines())
    for i in range(num_sample):
        print(i,"/",num_sample,end='\r')
        pred_img, path = sess.run([pred,image_path])
        dest_path = os.path.join(args.test_dir, path.decode('UTF-8').split("/")[-1])
        cv2.imwrite(dest_path,pred_img.astype(np.uint8)[0])
        
    coord.request_stop()
    coord.join(stop_grace_period_secs=30)