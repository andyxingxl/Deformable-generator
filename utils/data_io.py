#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: andy
"""

import matplotlib.pyplot as plt
import subprocess
import matplotlib.gridspec as gridspec
import tensorflow as tf

def read_and_decode(filename,shape):
      tfrecord_file_queue = tf.train.string_input_producer(filename, name='queue')
      reader = tf.TFRecordReader()
      _, tfrecord_serialized = reader.read(tfrecord_file_queue)
      tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'label': tf.FixedLenFeature([], tf.int64),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
      image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
      image = tf.reshape(image, shape)
      image = tf.cast(image, tf.float32) * (1. / 255)
      label = tfrecord_features['label']
      label = tf.cast(label, tf.int32)    
      return  image, label
  
def imnormalzo(image):
    immin=(image).min()
    immax=(image).max()
    image=(image-immin)/(immax-immin+1e-8)
    return image

def saveasimagesandvideo(samples,out_dir,ffmpeg_loglevel='quiet', fps=5):
    [num_frames, image_size, _, channel] = samples.shape
    for ifr in range(num_frames):
        fig,ax = plt.subplots()
        ax.imshow(imnormalzo(samples[ifr]), aspect='equal')
        fig = plt.gcf()
        fig.set_size_inches(image_size/96.0,image_size/96.0)
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.margins(0,0)
        plt.savefig("%s/%03d.png" % (out_dir, ifr),dpi=96.0,pad_inches=0.0)
        plt.close(fig) 
    '''
    # video
    subprocess.call('ffmpeg -loglevel {} -r {} -i {}/%03d.png -vcodec mpeg4 -y {}/sample.avi'.format(
            ffmpeg_loglevel, fps, out_dir, out_dir), shell=True)
    '''

def plotimgs(samples,Nh,Nc,IMG_HEIGHT,IMG_WIDTH,channel):
    fig = plt.figure(figsize=(Nc, Nh))
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.01, hspace=0.01)

    for i, sample in enumerate(samples[0:Nh*Nc,:,:,:]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
        image=imnormalzo(image)
        plt.imshow(image)
    return fig 
     