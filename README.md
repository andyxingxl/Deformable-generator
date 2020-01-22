## Code for the Deformable-Generator used in the following paper:

* "Deformable Generator Network: Unsupervised Disentanglement of Appearance and Geometry" by Xianglei Xing, Ruiqi Gao, Tian Han, Song-Chun Zhu, Ying Nian Wu.
* "Unsupervised Disentangling of Appearance and Geometry by Deformable Generator Network" by Xianglei Xing, Tian Han, Ruiqi Gao, Song-Chun Zhu, Ying Nian Wu.

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
