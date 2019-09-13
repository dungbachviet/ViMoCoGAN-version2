"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    saveVideos.py [options] <checkpoint_path>

Options:
    --resume                        when specified instance noise is used [default: False]
    --image_batch=<count>           number of images in image batch [default: 40]
    --video_batch=<count>           number of videos in video batch [default: 3]
    --factor=<count>                factor of total videos
    --dim_z_view=<count>            number of views
    --dim_z_category=<count>        number of categories
    --video_length=<count>          frames per video
    --input_path=<path>             specifies a path to a video or a image [default: ]
    --image_size=<count>            Size of image
"""


import os
import time
import sys
import shutil
import functools
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
import matplotlib.pyplot as plt
import PIL
import docopt
from torchvision import transforms


args = docopt.docopt(__doc__) # __doc__ is the first above comment of this file
print(args) # Display the parameters configuration

# checkpoint = torch.load("./log_folder/checkpoint_002023_000045.pth.tar")
checkpoint = torch.load(str(args['<checkpoint_path>']))

print("checkpoint['batch_num'] = ", checkpoint['batch_num'])
print("epoch_num = ", checkpoint["epoch_num"])
print("checkpoint['generator_loss'] = ", checkpoint['generator_loss'])
print("checkpoint['image_discriminator_loss'] = ", checkpoint['image_discriminator_loss'])
print("checkpoint['video_discriminator_loss'] = ", checkpoint['video_discriminator_loss'])

# Transform video (frames in one video)
def video_transform(video, image_transform):
    vid = [] # List containing tensor of image

    # each im (one frame) in video has dimension of (height, width, channel)
    # image_transform(im) ==> ouput : (channel, height, width) (normalized into -1 --> 1)
    # Create list vid = [(channel, height, width), (channel, height, width), (channel, height, width),...]
    for im in video:
        vid.append(image_transform(im))

    # Convert list of tensors into tensor
    # torch.stack(vid) (default dim=0) ==> (video_len, channel, height, width)
    # Continually, .permute(1, 0, 2, 3) ==> (channel, depth=video_len, height, width)
    vid = torch.stack(vid).permute(1, 0, 2, 3)
    return vid


image_transforms = transforms.Compose([
    PIL.Image.fromarray, # Convert numpy to PIL
    transforms.Scale((int(args["--image_size"]), int(args["--image_size"]))), # Scale to (image_size*height/width, image_size)
    # transforms.Scale((int(args["--image_size"]), int(args["--image_size"]))),
    transforms.ToTensor(), # Convert PIL to Tensor (0-->1) with dimension (channel, height, width)
    lambda x: x[:3, ::], # Only take out 3 channels (cause maybe there are images having 4 channels)
    transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)), # Normalize on each channels (3 channels) with mean[channel] and std[channel] correspondingly
])


video_transforms = functools.partial(video_transform, image_transform=image_transforms)
factor = int(args["--factor"])
dim_z_view = int(args["--dim_z_view"])
dim_z_category = int(args["--dim_z_category"])
num_class = dim_z_view * dim_z_category

video_length = int(args["--video_length"])
input_path = str(args["--input_path"])



# Create object of VideoGenerator
generator = models.VideoGenerator(n_channels=3, dim_z_content=50, dim_z_view=dim_z_view, dim_z_motion=10, dim_z_category=dim_z_category,
                dim_z_object=2, video_length=video_length, video_transforms=video_transforms, image_transforms=image_transforms, ngf=64)
print(generator)
print("torch.cuda.is_available() = ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.is_available() = ", torch.cuda.is_available())
    generator.cuda()

generator.load_state_dict(checkpoint['generator'])
generator.eval()

output_folder = "./test_encode_module"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Generate videos from one image
videos, z_view_labels, z_category_labels = generator.test_sample_videos_from_image(factor, input_path, video_length)
videos = videos_to_numpy(videos)

print("range(videos.size(0)) = ", range(videos.shape[0]))
for video_id in range(videos.shape[0]):
    # ==> (depth, height, width, channel) ==> Split into each image to display
    video = videos[video_id, ::].transpose((1, 2, 3, 0))
    turn = video_id//num_class
    print("turn = ", turn)
    for image_index in range(video.shape[0]):
        image = PIL.Image.fromarray(video[image_index, ::])

        save_directory = os.path.join(output_folder,
                                      "%04d_%04d/%04d" %
                                      (z_view_labels.data[video_id]+1, z_category_labels.data[video_id]+1, turn))
        print("save_directory = ", save_directory)

        if (not os.path.exists(save_directory)):
            os.makedirs(save_directory)
        image.save(os.path.join(save_directory, 'out%002d_%002d.jpg' % (video_id, image_index)))


