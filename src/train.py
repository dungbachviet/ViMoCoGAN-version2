#!/usr/bin/python
# -*- coding: latin-1 -*-
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    train.py [options] <dataset> <log_folder>

Options:
    --checkpoint_path=<path>        specifies a path to checkpoint file [default: ]
    --resume                        when specified instance noise is used [default: False]
    --image_dataset=<path>          specifies a separate dataset to train for images [default: ]
    --image_batch=<count>           number of images in image batch [default: 10]
    --video_batch=<count>           number of videos in video batch [default: 3]

    --image_size=<int>              resize all frames to this size [default: 64]

    --use_infogan                   when specified infogan loss is used

    --use_categories                when specified ground truth categories are used to
                                    train CategoricalVideoDiscriminator

    --use_noise                     when specified instance noise is used
    --noise_sigma=<float>           when use_noise is specified, noise_sigma controls
                                    the magnitude of the noise [default: 0]

    --image_discriminator=<type>    specifies image disciminator type (see models.py for a
                                    list of available models) [default: PatchImageDiscriminator]

    --video_discriminator=<type>    specifies video discriminator type (see models.py for a
                                    list of available models) [default: CategoricalVideoDiscriminator]

    --video_length=<len>            length of the video [default: 16]
    --log_interval=<count>          save checkpoint after each interval [default: 1000]
    --n_channels=<count>            number of channels in the input data [default: 3]
    --every_nth=<count>             sample training videos using every nth frame [default: 4]
    --batches=<count>               specify number of batches to train [default: 100000]

    --dim_z_content=<count>         dimensionality of the content input, ie hidden space [default: 50]
    --dim_z_motion=<count>          dimensionality of the motion input [default: 10]
    --dim_z_category=<count>        dimensionality of categorical input [default: 12]
    --dim_z_view=<count>            dimensionality of categorical input [default: 5]
    --dim_z_object=<count>          dimensionality of object input [default: 2]
"""

import os
import docopt
import PIL
from PIL import Image

import functools

import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader
from torchvision import transforms

import models
# from src import models

from trainers import Trainer
# from src.trainers import Trainer

import data
# from src import data

# Creat an object of network discriminator (expected)
# **kwargs : syntax to allow taking in many parameters of formats : key1=value, key2=value,...
# Inside the funtions, kwargs mean dictionary, *kwargs means set of keyword, **kwars means format of parameters
# type : string of type of discriminator object to create
def build_discriminator(type, **kwargs):
    discriminator_type = getattr(models, type) # Get Class (its name is of 'type') in module models (models.py)

    # # Check the correctness of parameters taken in
    # if 'Categorical' not in type and 'dim_categorical' in kwargs:
    #     kwargs.pop('dim_categorical') # ??? ?ây hình nh? là thao tác xóa

    return discriminator_type(**kwargs)# an object of Discriminator class expected


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


args = docopt.docopt(__doc__)
# Set up "main" to run this program
if __name__ == "__main__":
    # args = docopt.docopt(__doc__) # __doc__ is the first above comment of this file
    print(args) # Display the parameters configuration

    n_channels = int(args['--n_channels']) # Get number of channels of original images/videos

    # Set up a transform to images
    image_transforms = transforms.Compose([
        PIL.Image.fromarray, # Convert numpy to PIL
        transforms.Scale((int(args["--image_size"]), int(args["--image_size"]))), # Scale to (image_size*height/width, image_size)
        # transforms.Scale((int(args["--image_size"]), int(args["--image_size"]))),
        transforms.ToTensor(), # Convert PIL to Tensor (0-->1) with dimension (channel, height, width)
        lambda x: x[:n_channels, ::], # Only take out 3 channels (cause maybe there are images having 4 channels)
        transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)), # Normalize on each channels (3 channels) with mean[channel] and std[channel] correspondingly
    ])

    # Create "partial" funtions
    # video_transform is the name of function be partial
    # image_transform : is the one of two parameters of video_transform
    # The second parameter can be taken in the "partial" funtion after that
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    video_length = int(args['--video_length']) # Fixed number of frames in one video
    image_batch = int(args['--image_batch']) # batchsize of images
    video_batch = int(args['--video_batch']) # batchsize of videos

    dim_z_content = int(args['--dim_z_content']) # number of element in vector z_content
    dim_z_motion = int(args['--dim_z_motion']) #number of element in vector z_motion
    dim_z_category = int(args['--dim_z_category']) # number of element in vector z_category
    dim_z_view = int(args['--dim_z_view'])
    dim_z_object = int(args['--dim_z_object'])


    # Dataset of all
    dataset = data.VideoFolderDataset(args['<dataset>'], cache=os.path.join(args['<dataset>'], 'local.db'))

    # Object to get images from dataset VideoFolderDataset above
    image_dataset = data.ImageDataset(dataset, image_transforms)
    # Dataloader to load images
    image_loader = DataLoader(image_dataset, batch_size=image_batch, drop_last=True, num_workers=2, shuffle=True)

    # Object to get videos from dataset VideoFolderDataset above
    video_dataset = data.VideoDataset(dataset, video_length=16, every_nth=2, transform=video_transforms)
    # Dataloader to load videos
    video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=2, shuffle=True)

    # Object to get videos from dataset VideoFolderDataset above
    encode_video_dataset = data.VideoDataset(dataset, video_length=16, every_nth=2, transform=video_transforms)
    # Dataloader to load videos
    encode_video_loader = DataLoader(encode_video_dataset, batch_size=video_batch, drop_last=True, num_workers=2, shuffle=True)

    # Object to get images from dataset VideoFolderDataset above
    encode_image_dataset = data.ImageDataset(dataset, image_transforms)
    # Dataloader to load images
    encode_image_loader = DataLoader(encode_image_dataset, batch_size=image_batch, drop_last=True, num_workers=2, shuffle=True)





    # Create object of VideoGenerator
    generator = models.VideoGenerator(n_channels, dim_z_content=dim_z_content, dim_z_view=dim_z_view,
                                      dim_z_motion=dim_z_motion, dim_z_category=dim_z_category,
                                      dim_z_object=dim_z_object,
                                      video_length=video_length, encode_video_loader=encode_video_loader,
                                      encode_image_loader=encode_image_loader, video_transforms=video_transforms,
                                      image_transforms=image_transforms)

    # Create object of ImageDiscriminator
    image_discriminator = build_discriminator(args['--image_discriminator'], n_channels=n_channels,
                                              dim_z_view=dim_z_view,
                                              dim_z_object=dim_z_object,
                                              use_noise=args['--use_noise'],
                                              noise_sigma=float(args['--noise_sigma']))

    # Create object of VideoDiscriminator
    video_discriminator = build_discriminator(args['--video_discriminator'], n_channels=n_channels,
                                              dim_z_view=dim_z_view, dim_z_category=dim_z_category,
                                              dim_z_object=dim_z_object,
                                              use_noise=args['--use_noise'],
                                              noise_sigma=float(args['--noise_sigma']))

    # Push all networks to GPU
    print("torch.cuda.is_available() = ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.is_available() = ", torch.cuda.is_available())
        generator.cuda()
        image_discriminator.cuda()
        video_discriminator.cuda()

    # Object Trainer contains useful methods fro training
    # image_loader : Dataloader of batchsize of images
    # video_loader : Dataloader of batchsize of videos
    # args['--print_every'] : Display after each certain iteration
    # args['--batches'] : total batches in training process
    # use_cuda : boolean to indicate whether to use gpu
    # use_infogan : boolean to indicate whether to use InfoGan to calculate loss of category in Generator
    # use_categories : boolean to indicate whether to use category
    trainer = Trainer(image_loader, video_loader,
                      int(args['--log_interval']),
                      int(args['--batches']),
                      args['<log_folder>'],
                      checkpoint_path=args['--checkpoint_path'],
                      use_cuda=torch.cuda.is_available(),
                      use_infogan=args['--use_infogan'],
                      use_categories=args['--use_categories'],
                      resume=args['--resume']
                      )

    # Training to all 3 networks
    trainer.train(generator, image_discriminator, video_discriminator)




