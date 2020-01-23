#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:31:09 2020

@author: andy
"""

from utils.operations import warpnn
import numpy as np
import os
import tensorflow as tf
from utils.data_io import plotimgs
import matplotlib.pyplot as plt
import time


class DeformGenerator(object):
    def __init__(self, App_net,Geo_net,img_batch,imgsize,nchan,batchsz,za_dim,zg_dim,
            lamda,sigmap,lrp,step,step_size,N,Nepoch):
        self.App_net=App_net
        self.Geo_net=Geo_net
        self.img_batch=img_batch
        self.imgsize=imgsize
        self.nchan=nchan
        self.batchsz=batchsz
        self.za_dim=za_dim
        self.zg_dim=zg_dim
        self.lamda=lamda
        self.sigmap=sigmap
        self.step=step
        self.delta=step_size
        
        self.zapp =  tf.placeholder(shape=[None, self.za_dim], dtype=tf.float32, name='zapp')
        self.zgeo =  tf.placeholder(shape=[None, self.zg_dim], dtype=tf.float32, name='zgeo')
        self.y = tf.placeholder(tf.float32, shape = [None, imgsize,imgsize,nchan])
        self.lr=tf.placeholder(dtype=tf.float32,shape=())
        self.sigma=tf.placeholder(dtype=tf.float32,shape=())         
        self.N = N
        self.Nepoch=Nepoch       
        self.build_model()       
        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)
        self.saver = tf.train.Saver(max_to_keep=50)
    def build_model(self):
        self.gx = self.App_net(self.zapp)
        self.gdf = self.Geo_net(self.zgeo)
        self.gy = warpnn(self.gx, self.lamda*self.gdf,self.batchsz)
        self.loss = tf.nn.l2_loss(self.gy - self.y)/self.batchsz/tf.pow(self.sigma,2)
        self.lossr = tf.nn.l2_loss(self.gy - self.y)/self.batchsz*2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) \
                .minimize(self.loss, var_list=self.App_net.vars+self.Geo_net.vars)
        self.zapp_infer,self.zgeo_infer = self.langevin_infer(self.zapp,self.zgeo) 
    def langevin_infer(self,zapp,zgeo):
        def cond(i, zapp,zgeo):
             return tf.less(i, self.step)
        def body(i, zapp,zgeo):
            noise1 = tf.random_normal(shape=[self.batchsz, self.za_dim], name='noise1')
            noise2 = tf.random_normal(shape=[self.batchsz, self.zg_dim], name='noise2')
            gx = self.App_net(zapp,reuse=True)
            gdf = self.Geo_net(zgeo,reuse=True) 
            gy = warpnn(gx,self.lamda*gdf,self.batchsz)
            #ygenf =  refine(gx,Gy)
            loss = (tf.nn.l2_loss(gy - self.y)/tf.pow(self.sigma,2)+1e2*tf.nn.l2_loss(zapp)
                    +1e2*tf.nn.l2_loss(zgeo))/self.batchsz               
            grad1 = tf.gradients(loss, zapp, name='grad_genapp')[0]
            zapp = zapp - 0.5 * tf.pow(self.delta, 2) * grad1 + self.delta * noise1
            grad2 = tf.gradients(loss, zgeo, name='grad_gengeo')[0]
            zgeo = zgeo - 0.5 * tf.pow(self.delta, 2) * grad2 + self.delta * noise2
            return tf.add(i, 1), zapp,zgeo
        with tf.name_scope("langevin_infer"):
            i = tf.constant(0)
            i, zapp,zgeo = tf.while_loop(cond, body, [i, zapp,zgeo])
            return zapp,zgeo

    def train(self,img_batch,save_dirs):
      with tf.Session() as self.sess:
        self.sess.run(tf.global_variables_initializer())                  
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.get_default_graph().finalize()
        Nbatch=int(self.N/self.batchsz)
        zappvall = np.random.randn(Nbatch,self.batchsz,self.za_dim)
        zgeovall = np.random.randn(Nbatch,self.batchsz,self.zg_dim)
        lossr_all = np.zeros((self.Nepoch+1),np.float32)
        logfile=save_dirs['log_dir']+'log'
        f=open(logfile,"a")  
        for ie in range(self.Nepoch+1):
            Recons_loss=0.
            start_time = time.time()
             #lr 
            if ie <= 100:
                lrp=1e-3
            elif ie >100 and ie <=200:
                lrp=8e-4
            elif ie >200 and ie <=300:
                lrp=6e-4
            elif ie >300 and ie <=400:
                lrp=5e-4
            elif ie >400 and ie <=500:
                lrp=4e-4
            elif ie >500 and ie <=700:
                lrp=3e-4
            elif ie >700 and ie <=900:
                lrp=2e-4
            elif ie >900:
                lrp=1e-4
            # sigma rate
            if ie <= 100:
                sigmap=0.15
            elif ie >100:
                sigmap=0.1
            for ib in range(Nbatch):
                yp = self.sess.run(img_batch)
                zappp = zappvall[ib,:,:]
                zgeop = zgeovall[ib,:,:]
                feed_dict={self.zapp: zappp,self.zgeo:zgeop, self.y: yp,self.sigma:sigmap}
                zapp_infer,zgeo_infer=self.sess.run([self.zapp_infer,self.zgeo_infer], feed_dict=feed_dict)
                zappvall[ib,:,:]=zapp_infer
                zgeovall[ib,:,:]=zgeo_infer
                _,lossrp = self.sess.run([self.optimizer, self.lossr], 
                feed_dict={self.y: yp, self.zapp: zapp_infer,self.zgeo:zgeo_infer,self.sigma:sigmap,self.lr:lrp})
                Recons_loss=Recons_loss+ lossrp/Nbatch
            end_time = time.time()
            print('Epoch #{:d},Recons_loss:{:.3f},time: {:.2f}s'.format(ie, Recons_loss,end_time - start_time))
            
            lossr_all[ie]=Recons_loss
            if ie % 10 == 0:
                print('Epoch #{:d},Recons_loss:{:.3f},time: {:.2f}s'.format(ie, Recons_loss,end_time - start_time),
                        file=f)
                feed_dict={self.zapp: zapp_infer,self.zgeo:zgeo_infer}
                imgrec,imgrecbf = self.sess.run([self.gy,self.gx],feed_dict=feed_dict) 
                feed_dict={self.zapp: np.random.randn(self.batchsz, self.za_dim),
                        self.zgeo:np.random.randn(self.batchsz, self.zg_dim)}
                sample,samplebf = self.sess.run([self.gy,self.gx],feed_dict=feed_dict)
                self.plot(imgrec,10,ie,save_dirs['sample_dir'],'recons')
                self.plot(imgrecbf,10,ie,save_dirs['sample_dir'],'appout')
                self.plot(sample,10,ie,save_dirs['sample_dir'],'sampling')
                self.plot(samplebf,10,ie,save_dirs['sample_dir'],'sampappout')
            if ie % 50 == 0:
                self.save(ie,save_dirs)  
                self.savelatentvar(ie,zappvall,'zappvall',save_dirs)
                self.savelatentvar(ie,zgeovall,'zgeovall',save_dirs)
        coord.request_stop()
        coord.join(threads)
               
    def plot(self,img,ncol,ie,save_dirs,name):
        fig=plotimgs(img,int(self.batchsz/10),ncol,self.imgsize, self.imgsize,self.nchan)
        name='ep{}{}.png'.format(str(ie).zfill(3),name)
        plt.savefig(save_dirs+name, bbox_inches='tight')
        plt.close(fig)           
    def save(self,epoch,save_dirs):
        modelpath=save_dirs['params_dir']+'models'
        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
        self.saver.save(self.sess, modelpath+'/model', global_step=epoch,write_meta_graph=False) 
    def savelatentvar(self,epoch,var,varname,save_dirs):
        paramspath=save_dirs['params_dir']+'latentvars/'
        if not os.path.exists(paramspath):
            os.makedirs(paramspath)
        aa=''
        varname=[paramspath,varname,str(epoch).zfill(3)]
        np.save(aa.join(varname),var)
    def load(self,epoch,save_dirs):
        print('Loading the Model of epoch-{}...'.format(str(epoch)))
        modelpath=save_dirs['params_dir']+'models'
        self.saver.restore(self.sess, modelpath+'/model-'+str(epoch))
        print('Restored model parameters.')
    def evals(self,save_dirs):
        evalpath=save_dirs['eval_dir']
        if not os.path.exists(evalpath):
            os.makedirs(evalpath)
        npica=int(self.za_dim/10)
        npicg=int(self.zg_dim/10)
       
        '''
        (1) Plot the appearance basis functions
        '''
        for pic in range(npica):
            zgeop=np.zeros((self.batchsz, self.zg_dim))   
            zappp=np.zeros((self.batchsz, self.za_dim))
            for d in range(10):
                zappp[d*10:d*10+10,pic*10+d]=np.linspace(-10,10,10)      
            samples = self.sess.run(self.gy,feed_dict={self.zapp: zappp,self.zgeo:zgeop})           
            self.plot(samples,10,pic,save_dirs['eval_dir'],'appbasis')
      
        '''
        (2) Plot the geometric basis functions
        '''
        zappp =np.zeros((self.batchsz, self.za_dim))
        zappp[:,1]=np.zeros((self.batchsz)) - 8
        for pic in range(npicg):
            zgeop=np.zeros((self.batchsz, self.zg_dim))   
            for d in range(10):
                zgeop[d*10:d*10+10,pic*10+d]=np.linspace(-8,8,10)      
            samples = self.sess.run(self.gy,feed_dict={self.zapp: zappp,self.zgeo:zgeop})
            self.plot(samples,10,pic,save_dirs['eval_dir'],'geobasis')
           
     



            

            









        
