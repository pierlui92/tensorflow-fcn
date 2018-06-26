import tensorflow as tf
import os
import numpy as np

def build_input(input_list_txt):
    def _parse_function(image_tensor):
        image = tf.read_file(image_tensor[0])
        image_sem = tf.read_file(image_tensor[1])
        image = tf.image.decode_image(image, channels = 3)
        image_sem = tf.image.decode_image( image_sem , channels = 1)
        image.set_shape([None,None,3])
        image_sem.set_shape([None,None,1])
        image = tf.cast(image,tf.float32)
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