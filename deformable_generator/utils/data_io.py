#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: andy
"""

import matplotlib.pyplot as plt
import subprocess
import matplotlib.gridspec as gridspec

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
    fig = plt.figure(figsize=(10, 4))
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
     