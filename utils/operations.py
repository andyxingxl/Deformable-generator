import tensorflow as tf
import numpy as np

'''
Geometric warping
'''
def _meshgrid(height, width,nb):
        with tf.variable_scope('mymeshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            #ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat])
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([nb]))
            grid = tf.reshape(grid, tf.stack([nb, 2, -1]))
            return grid
        
def _repeat(x, n_repeats):
     with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

def _interpolate(im, x, y, xd,yd,out_size):
     with tf.variable_scope('myinterpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            xd = tf.cast(xd, 'float32')
            yd = tf.cast(yd, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
           
            zero1 = tf.zeros([], dtype='float32')
            max_y1 = tf.cast(tf.shape(im)[1] - 1, 'float32')
            max_x1 = tf.cast(tf.shape(im)[2] - 1, 'float32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0
            ###################
            x=x+xd
            y=y+yd
            x = tf.clip_by_value(x, zero1+0.000001, max_x1-0.000001)
            y = tf.clip_by_value(y, zero1+0.000001, max_y1-0.000001)
            ##################
            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output
        
def warpnn(im,dlm,nb): 
     imgsize=im.shape[1].value 
     nc=im.shape[3].value 
     lm=_meshgrid(imgsize, imgsize,nb)
     out_size =(imgsize,imgsize)
     x_s = tf.slice(lm, [0, 0, 0], [-1, 1, -1])
     y_s = tf.slice(lm, [0, 1, 0], [-1, 1, -1])
     x_s_flat = tf.reshape(x_s, [-1])
     y_s_flat = tf.reshape(y_s, [-1])
     xd_s = tf.slice(dlm, [0, 0, 0], [-1, 1, -1])
     yd_s = tf.slice(dlm, [0, 1, 0], [-1, 1, -1])
     xd_s_flat = tf.reshape(xd_s, [-1])
     yd_s_flat = tf.reshape(yd_s, [-1])
     input_transformed = _interpolate(
               im, x_s_flat, y_s_flat,xd_s_flat,yd_s_flat,
                out_size)
     Yhat = tf.reshape(
                input_transformed, tf.stack([nb, imgsize, imgsize, nc]))
     return Yhat

'''
 Basic operations in the networks
'''
def get_weight(shape, gain=np.sqrt(2), use_wscale=True,lmda=1., fan_in=None):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    #print( "current", shape[:-1], fan_in)
    std = gain / np.sqrt(fan_in) # He init
    std1= 0.1 # balance init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale * lmda
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.truncated_normal(stddev=std1))

# Fully-connected layer.
def dense(x, fmaps, gain=np.sqrt(2), use_wscale=True,lmda=1.,name='fc'):
  with tf.variable_scope(name):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale,lmda=lmda)
    w = tf.cast(w, x.dtype)
    bias = tf.get_variable("bias", [fmaps], initializer=tf.initializers.truncated_normal(stddev=0.1))
    return tf.matmul(x, w)+ bias


def deconv2d(x, fmaps,k=3, s=2, gain=np.sqrt(2), use_wscale=True, lmda=1.,padding='SAME', name="deconv2d"):
    with tf.variable_scope(name):
        w = get_weight([k, k, x.shape[-1].value,fmaps], gain=gain, use_wscale=use_wscale,lmda=lmda)
        w = tf.cast(w, x.dtype)
        w = tf.transpose(w,[0,1,3,2])
        if s == 2:
            output_shape = [tf.shape(x)[0],s*x.shape[1].value,s*x.shape[2].value,fmaps]  
        else: 
            output_shape = [tf.shape(x)[0],s,s,fmaps] 
        conv = tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape, strides=[1, s, s, 1], padding=padding)
        biases = tf.get_variable('biases', [fmaps], initializer=tf.initializers.truncated_normal(stddev=0.1))
        conv = tf.nn.bias_add(conv, biases)      
        return conv


