#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo 

@author: andyxing
"""
import tensorflow as tf
import numpy as np
import os
import argparse
import importlib
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from model import DeformGenerator
from utils.data_io import read_and_decode
tf.reset_default_graph()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='./dataset/celebacrop10k.tfrecords')
    parser.add_argument('--netst', type=str, default='balsch1')
    parser.add_argument('--zad', type=int, default=64)
    parser.add_argument('--zgd', type=int, default=64)
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--imgsize', type=float, default=64)
    parser.add_argument('--nchan', type=float, default=3)
    parser.add_argument('--kersize', type=list, default=[3,3,5,5])
    parser.add_argument('--fmd', type=list, default=[4,8,16,32,64])
    parser.add_argument('--chb', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=5/8)
    parser.add_argument('--lamda', type=float, default=20.0)
    parser.add_argument('--sigmap', type=float, default=0.1)
    parser.add_argument('--lrp', type=float, default=0.7)
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--Nepoch', type=int, default='1000')
    parser.add_argument('--train', type=str, default='True')
   

    args = parser.parse_args()
    imgsize = args.imgsize
    nchan = args.nchan
    batchsz = args.bs
    kersize =  args.kersize
    chb = args.chb
    fmd = args.fmd
    netstruct = importlib.import_module('netstructs'+ '.' + args.netst)
    seed = 7;np.random.seed(seed);tf.set_random_seed(seed)
    App_net =  netstruct.app_net(za_dim=args.zad,nchan=nchan,imgsize=imgsize,batchsize=batchsz,
            kersize=kersize,chb=chb,alpha=args.alpha,fmd=fmd)
    Geo_net =  netstruct.geo_net(zg_dim=args.zgd,nchan=nchan,imgsize=imgsize,batchsize=batchsz,
            kersize=kersize,chb=chb,fmd=fmd)
    
    tfrecords=args.data
    images, _ = read_and_decode([tfrecords],[imgsize,imgsize,nchan])
    img_batch = tf.train.batch([images],batch_size=batchsz, 
                                            capacity=batchsz * 5)
    
    DeformGen = DeformGenerator(App_net,Geo_net,img_batch,imgsize,nchan,batchsz,
             args.zad, args.zgd, args.lamda,args.sigmap,args.lrp,args.step,args.step_size,args.N,args.Nepoch)
    dir = os.path.dirname(os.path.realpath(__file__))
    log_dir=dir+'/log/'
    params_dir=dir+'/params/'
    sample_dir=dir+'/figrecs/'
    eval_dir=dir+'/figeval/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    save_dirs = {'log_dir': log_dir, 'params_dir': params_dir, 'sample_dir': sample_dir,'eval_dir':eval_dir}
    if args.train == 'True':
        DeformGen.train(img_batch,save_dirs)
    else:
        print('Attempting to Restore Model ...')
        DeformGen.load(args.Nepoch,save_dirs)
        DeformGen.evals(save_dirs)

