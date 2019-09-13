"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    saveLoss.py <checkpoint_path>

"""


import os
import time
import sys
import shutil

import numpy as np

import torch

torch.backends.cudnn.enabled = False
from torch import nn

from torch.autograd import Variable
import torch.optim as optim

import os
from trainers import videos_to_numpy
import subprocess as sp
import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL


import docopt


args = docopt.docopt(__doc__) # __doc__ is the first above comment of this file
print(args) # Display the parameters configuration

checkpoint = torch.load(args['<checkpoint_path>'])
batch_num = (checkpoint['batch_num'] + 1)  # cause it's the new iteration
generator_loss = checkpoint['generator_loss']
image_discriminator_loss = checkpoint['image_discriminator_loss']
video_discriminator_loss = checkpoint['video_discriminator_loss']


print("checkpoint['batch_num'] = ", checkpoint['batch_num'])

print("checkpoint['generator_loss'] = ", checkpoint['generator_loss'])

print("checkpoint['image_discriminator_loss'] = ", checkpoint['image_discriminator_loss'])

print("checkpoint['video_discriminator_loss'] = ", checkpoint['video_discriminator_loss'])

print("epoch_num = ", checkpoint["epoch_num"])


output_folder = "./saveLoss"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

x = range(len(generator_loss))
plt.plot(x, video_discriminator_loss, 'bo-')
plt.plot(x, image_discriminator_loss, 'ro-')
plt.plot(x, generator_loss, 'yo-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("./saveLoss/loss.jpg")






