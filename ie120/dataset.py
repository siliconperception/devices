# download Imagenet dataset and generate flist files
import os
import argparse
import subprocess

os.chdir('..')
if not os.path.exists('imagenet'):
    os.makedirs('imagenet')
os.chdir('imagenet')
subprocess.run(["wget", "https://image-net.org/data/ILSVRC/2010/ILSVRC2010_images_train.tar"])
subprocess.run(["wget", "https://image-net.org/data/ILSVRC/2010/ILSVRC2010_devkit-1.0.tar.gz"])
subprocess.run(["wget", "https://image-net.org/data/ILSVRC/2010/ILSVRC2010_images_val.tar"])
subprocess.run(["gunzip", "ILSVRC2010_devkit-1.0.tar.gz"])
subprocess.run(["tar", "-xf", "ILSVRC2010_images_train.tar"])
subprocess.run(["tar", "-xf", "ILSVRC2010_devkit-1.0.tar"])
subprocess.run(["tar", "-xf", "ILSVRC2010_images_val.tar"])
train_flist = subprocess.run(["ls", "-1", "train"], capture_output=True).stdout
with open("train/flist","w") as f:
    print(train_flist,file=f)
val_flist = subprocess.run(["ls", "-1", "val"], capture_output=True).stdout
with open("val/flist","w") as f:
    print(val_flist,file=f)
