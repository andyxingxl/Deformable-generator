#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo for (1) obtaining the typical appearance and geometric basis functions 
             from the pre-trained models.
         (2) applying the viewing angle and shape basis fucntions to warp the 
             appearance basis functions to generate new images. 

@author: andy
"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from utils.operations import warpnn
from utils.data_io import saveasimagesandvideo
from utils.data_io import plotimgs
import matplotlib.pyplot as plt
tf.reset_default_graph()

dir = os.path.dirname(os.path.realpath(__file__))
paramspath=dir+'/paramsraw/'
figpathapp=dir+'/savetypicalbasis_app/'
figpathgeo=dir+'/savetypicalbasis_geo/'
figpathwarp=dir+'/savewarpingresults/'
modelname='model.ckpt-1000'
modelpath=paramspath+modelname
varname1='zappvall'
varname2='zgeovall'

batchsize=100
imgsize = 64
channels=3
z_app = 64
z_geo = 64

kersize=[3,3,5,5]
chb=16
channelg=[chb*8,chb*4,chb*2,chb*1]
alpha=5/8
channela=[int(i*alpha) for i in channelg]
fmd=[4,8,16,32,64]


Y = tf.placeholder(tf.float32, shape = [None, imgsize,imgsize,channels])
zapp = tf.placeholder(shape=[None, z_app], dtype=tf.float32, name='zapp')
zgeo = tf.placeholder(shape=[None, z_geo], dtype=tf.float32, name='zgeo')
weightsG = {
    'wgaf1': tf.Variable(tf.truncated_normal([z_app, fmd[0]*fmd[0]*channela[0]],stddev = 0.1)), 
    'wgac1': tf.Variable(tf.truncated_normal([kersize[0], kersize[0], channela[1], channela[0]],stddev = 0.1)),  
    'wgac2': tf.Variable(tf.truncated_normal([kersize[1], kersize[1], channela[2], channela[1]],stddev = 0.1)),  
    'wgac3': tf.Variable(tf.truncated_normal([kersize[2], kersize[2], channela[3], channela[2]],stddev = 0.1)),  
    'wgac4': tf.Variable(tf.truncated_normal([kersize[3], kersize[3], 3, channela[3]],stddev = 0.1)),  
   ###
    'wggf1': tf.Variable(tf.truncated_normal([z_geo, fmd[0]*fmd[0]*channelg[0]],stddev = 0.1)), 
    'wggc1': tf.Variable(tf.truncated_normal([kersize[0], kersize[0], channelg[1], channelg[0]], stddev = 0.1)),
    'wggc2': tf.Variable(tf.truncated_normal([kersize[1], kersize[1], channelg[2], channelg[1]],stddev = 0.1)),
    'wggc3': tf.Variable(tf.truncated_normal([kersize[2], kersize[2], channelg[3], channelg[2]],stddev = 0.1)),
    'wggc4': tf.Variable(tf.truncated_normal([kersize[3], kersize[3], 2, channelg[3]],stddev = 0.1))
  }
biasesG = {   
    'bgaf1': tf.Variable(tf.truncated_normal([fmd[0]*fmd[0]*channela[0]], stddev = 0.1)),
    'bgac1': tf.Variable(tf.truncated_normal([channela[1]], stddev = 0.1)),
    'bgac2': tf.Variable(tf.truncated_normal([channela[2]], stddev = 0.1)),
    'bgac3': tf.Variable(tf.truncated_normal([channela[3]], stddev = 0.1)),
    'bgac4': tf.Variable(tf.truncated_normal([3], stddev = 0.1)),
    'bggf1': tf.Variable(tf.truncated_normal([fmd[0]*fmd[0]*channelg[0]], stddev = 0.1)),
    'bggc1': tf.Variable(tf.truncated_normal([channelg[1]], stddev = 0.1)), 
    'bggc2': tf.Variable(tf.truncated_normal([channelg[2]], stddev = 0.1)), 
    'bggc3': tf.Variable(tf.truncated_normal([channelg[3]], stddev = 0.1)), 
    'bggc4': tf.Variable(tf.truncated_normal([2], stddev = 0.1)),          
  } 

def conv2d(x, W, b, strides=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')

def decoderapp(zapp,weights, biases):
     hp=tf.nn.relu(tf.matmul(zapp, weights['wgaf1']) + biases['bgaf1'])
     hc=tf.reshape(hp,[-1, fmd[0], fmd[0], channela[0]])    
     output_shape_da_conv1 = tf.stack([batchsize, fmd[1], fmd[1], channela[1]])
     output_shape_da_conv2 = tf.stack([batchsize, fmd[2], fmd[2], channela[2]])
     output_shape_da_conv3 = tf.stack([batchsize, fmd[3], fmd[3], channela[3]])
     output_shape_da_conv4 = tf.stack([batchsize, fmd[4], fmd[4], 3])
     h_d_conv1 = tf.nn.relu(deconv2d(hc, weights['wgac1'], output_shape_da_conv1)+ biases['bgac1'])
     h_d_conv2 = tf.nn.relu(deconv2d(h_d_conv1, weights['wgac2'], output_shape_da_conv2)+ biases['bgac2'])
     h_d_conv3 = tf.nn.relu(deconv2d(h_d_conv2, weights['wgac3'], output_shape_da_conv3)+ biases['bgac3'])     
     gx = tf.nn.sigmoid(deconv2d(h_d_conv3, weights['wgac4'], output_shape_da_conv4)+ biases['bgac4'])
     return gx
 
def decodergeo(zgeo,weights, biases):
     hp=tf.nn.relu(tf.matmul(zgeo, weights['wggf1']) + biases['bggf1'])
     hc=tf.reshape(hp,[-1, fmd[0], fmd[0], channelg[0]])
     output_shape_dg_conv1 = tf.stack([batchsize, fmd[1], fmd[1], channelg[1]])
     output_shape_dg_conv2 = tf.stack([batchsize, fmd[2], fmd[2], channelg[2]])
     output_shape_dg_conv3 = tf.stack([batchsize, fmd[3], fmd[3], channelg[3]])
     output_shape_dg_conv4 = tf.stack([batchsize, fmd[4], fmd[4], 2])
     h_d_conv1 = tf.nn.relu(deconv2d(hc, weights['wggc1'], output_shape_dg_conv1)+ biases['bggc1'])
     h_d_conv2 = tf.nn.relu(deconv2d(h_d_conv1, weights['wggc2'], output_shape_dg_conv2)+ biases['bggc2'])
     h_d_conv3 = tf.nn.relu(deconv2d(h_d_conv2, weights['wggc3'], output_shape_dg_conv3)+ biases['bggc3'])
     glm = tf.nn.tanh(deconv2d(h_d_conv3, weights['wggc4'], output_shape_dg_conv4)+ biases['bggc4'])
     glm = tf.reshape(glm,(-1,imgsize*imgsize,2)) 
     glm = tf.transpose(glm,[0,2,1])
     return glm
gx = decoderapp(zapp,weightsG, biasesG)
glm = decodergeo(zgeo,weightsG, biasesG) 
lamda=20.0   
Gy = warpnn(gx,lamda*glm,batchsize)

saver = tf.train.Saver()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, modelpath)
    if not os.path.exists(figpathapp):
        os.makedirs(figpathapp)
    if not os.path.exists(figpathgeo):
        os.makedirs(figpathgeo)
    if not os.path.exists(figpathwarp):
        os.makedirs(figpathwarp)           
    npica=int(z_app/10)
    npicg=int(z_geo/10)
    Yappball=np.zeros((npica,batchsize,imgsize, imgsize, channels))
    Ygeoball=np.zeros((npicg,batchsize,imgsize, imgsize, channels))
    
    
    '''
    (1.1) Plot the typical appearance basis functions
    '''
    zappp =np.zeros((batchsize, z_app))
    zgeop =np.zeros((batchsize, z_geo))
   
            
    for pic in range(npica):
        zgeop=np.zeros((batchsize, z_geo))   
        zappp=np.zeros((batchsize, z_app))
        for d in range(10):
            zappp[d*10:d*10+10,pic*10+d]=np.linspace(-10,10,10)      
        samples = sess.run(Gy,feed_dict={zapp: zappp,zgeo:zgeop})
        Yappball[pic]=samples
    Yappball=np.reshape(Yappball,[-1,10,imgsize, imgsize, channels])
    tabsis=[]
    taidlist=[49,25,21,9]
    for tid in range(len(taidlist)):
        ids=taidlist[tid]
        tabsis.append(Yappball[ids])
    tabsis=np.concatenate(tabsis,0)
    saveasimagesandvideo(tabsis,figpathapp) 
    fig=plotimgs(tabsis,4,10,imgsize, imgsize,channels)
    plt.savefig("%s.png" % (figpathapp+'appbasis'))
    plt.close(fig) 
       
    
    '''
    (1.2) Plot the typical geometric basis functions
    '''
    zappp =np.zeros((batchsize, z_app))
    zappp[:,1]=np.zeros((batchsize))-5
   
    for pic in range(npicg):
        zgeop=np.zeros((batchsize, z_geo))   
        for d in range(10):
            zgeop[d*10:d*10+10,pic*10+d]=np.linspace(-8,8,10)      
        samples = sess.run(Gy,feed_dict={zapp: zappp,zgeo:zgeop})
        Ygeoball[pic]=samples
    Ygeoball=np.reshape(Ygeoball,[-1,10,imgsize, imgsize, channels])
    tgbsis=[]
    tgidlist=[4,39,10,50]
    for tid in range(len(tgidlist)):
        ids=tgidlist[tid]
        tgbsis.append(Ygeoball[ids])
    tgbsis=np.concatenate(tgbsis,0)
    saveasimagesandvideo(tgbsis,figpathgeo) 
    fig=plotimgs(tgbsis,4,10,imgsize, imgsize,channels)
    plt.savefig("%s.png" % (figpathgeo+'geobasis'))
    plt.close(fig) 
    
    '''
    (2.1) applying the viewing angle basis fucntions to warp the 
             appearance basis functions to generate new images
    '''
    zappp =np.zeros((batchsize, z_app))
    zgeop =np.zeros((batchsize, z_geo))
    Yappball=np.zeros((npica,batchsize,imgsize, imgsize, channels))        
    for pic in range(npica):
        zgeop=np.zeros((batchsize, z_geo))   
        zappp=np.zeros((batchsize, z_app))
        for d in range(10):
            zgeop[d*10:d*10+10,4]=np.linspace(-8,8,10)  
        for d in range(10):
            zappp[d*10:d*10+10,pic*10+d]=np.linspace(-10,10,10)      
        samples = sess.run(Gy,feed_dict={zapp: zappp,zgeo:zgeop})
        Yappball[pic]=samples
    Yappball=np.reshape(Yappball,[-1,10,imgsize, imgsize, channels])
    tabsis=[]
    taidlist=[49,25,21,9]
    for tid in range(len(taidlist)):
        ids=taidlist[tid]
        tabsis.append(Yappball[ids])
    tabsis=np.concatenate(tabsis,0)
    fig=plotimgs(tabsis,4,10,imgsize, imgsize,channels)
    plt.savefig("%s.png" % (figpathwarp+'viewwarp'))
    plt.close(fig) 
    
    
    '''
    (2.2) applying the shape basis fucntions to warp the 
             appearance basis functions to generate new images
    '''
    zappp =np.zeros((batchsize, z_app))
    zgeop =np.zeros((batchsize, z_geo))
    Yappball=np.zeros((npica,batchsize,imgsize, imgsize, channels))        
    for pic in range(npica):
        zgeop=np.zeros((batchsize, z_geo))   
        zappp=np.zeros((batchsize, z_app))
        for d in range(10):
            zgeop[d*10:d*10+10,10]=np.linspace(-8,8,10)  
        for d in range(10):
            zappp[d*10:d*10+10,pic*10+d]=np.linspace(-10,10,10)      
        samples = sess.run(Gy,feed_dict={zapp: zappp,zgeo:zgeop})
        Yappball[pic]=samples
    Yappball=np.reshape(Yappball,[-1,10,imgsize, imgsize, channels])
    tabsis=[]
    taidlist=[49,25,21,9]
    for tid in range(len(taidlist)):
        ids=taidlist[tid]
        tabsis.append(Yappball[ids])
    tabsis=np.concatenate(tabsis,0)
    fig=plotimgs(tabsis,4,10,imgsize, imgsize,channels)
    plt.savefig("%s.png" % (figpathwarp+'shapewarp'))
    plt.close(fig) 