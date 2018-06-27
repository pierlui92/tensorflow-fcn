import numpy as np
import argparse
import os
import cv2
import tensorflow as tf
from models.fcn8_vgg import FCN8VGG
from collections import namedtuple
from utils import *

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (2550,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (1255, 11, 32) ),
]


trainId2name = { label.trainId : label.name for label in labels }

ignore_label = 255

parser = argparse.ArgumentParser(description='Evaluation on the cityscapes validation set')
parser.add_argument('--checkpoint_dir',  type=str,   help='folder containing checkpoints', required=True)
parser.add_argument('--gt_file', type=str,  help='path to filelist.txt', required=True)
parser.add_argument('--num_classes', type=int, default= 19, help='num classes')
parser.add_argument('--output_path', type=str, default='validation.txt')
args = parser.parse_args()


### INPUTS ###
image_placeholder = tf.placeholder(tf.float32)
sem_gt_placeholder = tf.placeholder(tf.int32)

input_images = tf.cast(tf.expand_dims(image_placeholder, axis=0),tf.float32)
sem_gt = tf.expand_dims(sem_gt_placeholder, axis=0)

with tf.name_scope("content_vgg"):
    vgg_fcn = FCN8VGG()
    vgg_fcn.build(input_images,train=False, debug=False, num_classes=args.num_classes)
    sem_pred = vgg_fcn.pred_up

### MIOU ###
weightsValue = tf.to_float(tf.not_equal(sem_gt,ignore_label))
sem_gt = tf.where(tf.equal(sem_gt, ignore_label), tf.zeros_like(sem_gt), sem_gt)
sem_pred = tf.where(tf.equal(sem_pred, ignore_label), tf.zeros_like(sem_pred), sem_pred)
miou, update_op = tf.metrics.mean_iou(labels=tf.reshape(sem_gt,[-1]),predictions=tf.reshape(sem_pred,[-1]), num_classes=args.num_classes, weights=tf.reshape(weightsValue,[-1]))

summary_miou = tf.summary.scalar("miou",miou)

print('Finished building Network.')

if not os.path.exists(os.path.join(args.checkpoint_dir,"val")):
    os.mkdir(os.path.join(args.checkpoint_dir,"val"))
writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir,"val"))

init = [tf.global_variables_initializer(),tf.local_variables_initializer()]

list_checkpoints = {}
while True:
    print("Waiting for new checkpoint", end='\r')
    best=0
    for f in sorted(os.listdir(args.checkpoint_dir)):
        if "fcn8s-" in f:
            output = open(args.output_path,"a")
            basename=f.split(".")[0]
            if basename not in list_checkpoints.keys():
                list_checkpoints[basename]=os.path.join(args.checkpoint_dir, basename)
                with tf.Session() as sess:
                    sess.run(init)
                    step = load(sess,list_checkpoints[basename])
                    
                    print("Loading last checkpoint")
                    if  step >= 0:
                        print("Restored step: ", step)
                        print(" [*] Load SUCCESS")
                    else:
                        step=0
                        print(" [!] Load failed...")   

                    coord = tf.train.Coordinator()
                    tf.train.start_queue_runners()
                    print('Thread running')
                    print('Running the Network')

                    lenght=len(open(args.gt_file).readlines())
                    with open(args.gt_file) as filelist:
                        for idx,line in enumerate(filelist):
                            print("Image evaluated: ",idx + 1,"/",lenght,end='\r')
                            image = cv2.imread(line.split(";")[0])
                            semgt = cv2.imread(line.split(";")[-1].strip(),cv2.IMREAD_GRAYSCALE)
                            _=sess.run(update_op,feed_dict={image_placeholder : image , sem_gt_placeholder : semgt})
                            miou_value =sess.run(miou,feed_dict={image_placeholder : image , sem_gt_placeholder : semgt})
                        sum_str = sess.run(summary_miou)
                        writer.add_summary(sum_str,step)

                    if miou_value > best:
                        output.write("!!!!!!!!NEW BEST!!!!!!!!\n")
                        best = miou_value
                    output.write("########" + str(step) + "########\n")      
                    mean=0
                    confusion_matrix=tf.get_default_graph().get_tensor_by_name("mean_iou/total_confusion_matrix:0").eval()
                    for cl in range(confusion_matrix.shape[0]):
                        tp_fn = np.sum(confusion_matrix[cl,:])
                        tp_fp = np.sum(confusion_matrix[:,cl])
                        tp = confusion_matrix[cl,cl]
                        IoU_cl = tp / (tp_fn + tp_fp - tp)
                        output.write(trainId2name[cl] + ": {:.4f}\n".format(IoU_cl))
                    output.write("#######################\n")
                    output.write("mIoU: " + str(miou_value) +"\n")
                    
                    coord.request_stop()
                    coord.join(stop_grace_period_secs=30)
            output.close()

