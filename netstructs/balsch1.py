import tensorflow as tf
from utils.operations import dense
from utils.operations import deconv2d

class app_net(object):
    def __init__(self,za_dim=64,nchan=3,imgsize = 64,batchsize=100,
            kersize=[3,3,5,5],chb=16,alpha=5/8,fmd=[4,8,16,32,64]):
        
        self.za_dim=za_dim
        self.nchan = nchan
        self.imgsize = imgsize
        self.bs = batchsize
        self.kersize=kersize
        self.chb=chb
        channelg=[chb*8,chb*4,chb*2,chb*1]
        self.alpha=alpha
        self.channela=[int(i*alpha) for i in channelg]      
        self.fmd = fmd
        self.name = 'genapp'
    def __call__(self, z, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            ch=self.fmd[0]*self.fmd[0]*self.channela[0]
            hf = dense(z, ch,use_wscale=False)
            hc=tf.nn.relu(tf.reshape(hf,[-1, self.fmd[0], self.fmd[0], self.channela[0]]))   # 4x4
            fmd1=tf.nn.relu(deconv2d(hc,self.channela[1],k=self.kersize[0],use_wscale=False,name='deconv1')) #8x8
            fmd2=tf.nn.relu(deconv2d(fmd1,self.channela[2],k=self.kersize[1],use_wscale=False,name='deconv2')) #16x16
            fmd3=tf.nn.relu(deconv2d(fmd2,self.channela[3],k=self.kersize[2],use_wscale=False,name='deconv3')) #32x32
            gx = tf.nn.sigmoid(deconv2d(fmd3,3,k=self.kersize[3],use_wscale=False,name='deconv4')) #64x64
            return gx
        
    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if var.name.startswith(self.name)]


class geo_net(object):
    def __init__(self,zg_dim=64,nchan=3,imgsize = 64,batchsize=100,
            kersize=[3,3,5,5],chb=16,fmd=[4,8,16,32,64]):
        
        self.zg_dim=zg_dim
        self.nchan = nchan
        self.imgsize = imgsize
        self.bs = batchsize
        self.kersize=kersize
        self.chb=chb
        self.channelg=[chb*8,chb*4,chb*2,chb*1]
        self.fmd = fmd
        self.name = 'gengeo'
    def __call__(self, z, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            ch=self.fmd[0]*self.fmd[0]*self.channelg[0]
            hf = dense(z, ch,use_wscale=False)
            hc=tf.nn.relu(tf.reshape(hf,[-1, self.fmd[0], self.fmd[0], self.channelg[0]]))   # 4x4
            fmd1=tf.nn.relu(deconv2d(hc,self.channelg[1],k=self.kersize[0],use_wscale=False,name='deconv1')) #8x8
            fmd2=tf.nn.relu(deconv2d(fmd1,self.channelg[2],k=self.kersize[1],use_wscale=False,name='deconv2')) #16x16
            fmd3=tf.nn.relu(deconv2d(fmd2,self.channelg[3],k=self.kersize[2],use_wscale=False,name='deconv3')) #32x32
            gdfq = tf.nn.tanh(deconv2d(fmd3,2,k=self.kersize[3],use_wscale=False,name='deconv4')) #64x64
            gdf = tf.reshape(gdfq,(-1,self.imgsize*self.imgsize,2))
            gdf = tf.transpose(gdf,[0,2,1])
            return gdf
        
    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
            

            
    

