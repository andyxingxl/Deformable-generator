## Code for the Deformable-Generator used in the following paper:

* "Deformable Generator Networks: Unsupervised Disentanglement of Appearance and Geometry" by Xianglei Xing, Ruiqi Gao, Tian Han, Song-Chun Zhu, Ying Nian Wu. [IEEE T-PAMI](https://ieeexplore.ieee.org/document/9158550),  [arXiv](https://arxiv.org/abs/1806.06298)
* "Unsupervised Disentangling of Appearance and Geometry by Deformable Generator Network" by Xianglei Xing, Tian Han, Ruiqi Gao, Song-Chun Zhu, Ying Nian Wu. [CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Xing_Unsupervised_Disentangling_of_Appearance_and_Geometry_by_Deformable_Generator_Network_CVPR_2019_paper.html)

## For more information, please visit the [Project Page](https://andyxingxl.github.io/Deformable-generator/).
## Requirements:
* TensorFlow (see http://www.tensorflow.org for how to install)
* Python, NumPy (see http://www.numpy.org/)
* argparse,matplotlib,importlib

## Quick test from pre-trained model of CelebA:
```bash
$ python3 demo_pretrain.py
```

## To train the model from scratch:
* Unzip 'celebacrop10k.7z' in the 'dataset' folder, and extract 'celebacrop10k.tfrecords' into 'dataset'.
* ``` $ python3 main_dfg.py```

## To test the model:
```bash
$ python3 main_dfg.py --train='False'
```

## Citation:
Please cite this paper in your publications if it helps your research:
```bash
@ARTICLE{9158550,  
author={Xing, Xianglei and Gao, Ruiqi and Han, Tian and Zhu, Song-Chun and Wu, Ying Nian},  
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},   
title={Deformable Generator Networks: Unsupervised Disentanglement of Appearance and Geometry},   
year={2020},  
pages={1-1},  
doi={10.1109/TPAMI.2020.3013905}
}
```
Link to paper:
* [Deformable Generator Networks: Unsupervised Disentanglement of Appearance and Geometry](https://ieeexplore.ieee.org/document/9158550)
