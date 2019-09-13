#!/usr/bin/python
# -*- coding: latin-1 -*-
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import os
import time
import shutil
import sys
import random
import PIL

import numpy as np

import torch
torch.backends.cudnn.enabled = False
from torch import nn

from torch.autograd import Variable
import torch.optim as optim


# Use GPU if computer supports Nvidia Screen Card
if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

# Convert Tensor of Image to Numpy
def images_to_numpy(tensor):
    # Tensor Image (batch_size, channels, height, width) --> Numpy Image (batch_size, height, width, channels)
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
    # Assign elements lesser than -1 to -1
    generated[generated < -1] = -1
    # Assign elements larger than 1 to 1
    generated[generated > 1] = 1
    # Normalize to domain : 0-->1
    generated = (generated + 1) / 2 * 255
    # Cast to type of unsigned integer 8 bytes
    return generated.astype('uint8')

# Convert from tensor of video to numpy
# Normalize to domain : 0-->1
def videos_to_numpy(tensor):
    # Convert Tensor Variable to numpy with dimensions : ??? (batch_size, channels, depth, height, width)
    generated = tensor.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


# Chỉ ra index khác 0 trong vector one-hot
def one_hot_to_class(tensor):
    a, b = np.nonzero(tensor)
    return np.unique(b).astype(np.int32)

# Design useful class containing neccessary functions for training
# image_sampler : DataLoader of Images (batchsize)
# video sampler : DataLoader of Videos (batchsize)
# log_interval : Display results after number of iterations
# train_batches : The total number of batches on training
# log_folder : Path to log file
# use_cuda : Use GPU or not
# use_infogan : Calculate loss of generator on fooling discriminator on action categories
# use_categories : Use categories or not
class Trainer(object):
    def __init__(self, image_sampler, video_sampler, log_interval, train_batches, log_folder,
                 checkpoint_path, use_cuda=False, use_infogan=True, use_categories=True, resume=False):

        self.use_categories = use_categories

        self.gan_criterion = nn.BCEWithLogitsLoss() # Thiết lâp hàm tính lỗi cho GAN
        self.category_criterion = nn.CrossEntropyLoss() # Thiết lập hàm tính lỗi cho category (để biết bộ học có sinh ra đúng category hay không? Giá trị hàm lỗi càng nhỏ thì càng tốt)

        self.image_sampler = image_sampler
        self.video_sampler = video_sampler

        # Batchsize of videos and images
        self.video_batch_size = self.video_sampler.batch_size
        self.image_batch_size = self.image_sampler.batch_size

        self.log_interval = log_interval
        self.train_batches = train_batches

        self.log_folder = log_folder

        self.use_cuda = use_cuda
        self.use_infogan = use_infogan

        self.checkpoint_path = checkpoint_path
        self.resume = resume

        # Wrap dataloader in enumeration
        self.image_enumerator = None
        self.video_enumerator = None

    # Create ones tensor
    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    # Create zeros tensors
    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)


    # Sample real batchsize of images
    def sample_real_image_batch(self):
        # Wrap the image_sampler (dataLoader for Images) into Enumerate
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.image_sampler)

        # Get the index of batchsize and batch of images
        batch_idx, batch = next(self.image_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.iteritems():
                b[k] = v.cuda()

        # If traversed to last batch ==> Set up new enumeration for next new epoch
        if batch_idx == len(self.image_sampler) - 1:
            self.image_enumerator = enumerate(self.image_sampler)

        return b # Return real batchsize of images (pushed to GPU if use it)


    # Sample a real batchsize of videos
    def sample_real_video_batch(self):
        # Wrap the video_sampler (dataLoader for Vides) into Enumerate (for first use time)
        if self.video_enumerator is None:
            self.video_enumerator = enumerate(self.video_sampler)

        # Get the index of batchsize and batch of videos
        batch_idx, batch = next(self.video_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.iteritems():
                b[k] = v.cuda()

        # If traversed to last batch ==> Set up new enumeration for next new epoch
        if batch_idx == len(self.video_sampler) - 1:
            self.video_enumerator = enumerate(self.video_sampler)

        return b # Return real batchsize of videos (pushed to GPU if use it)

    # Calculate loss for discrimator (on images or videos), optimsize the parameters to reduce loss
    # discriminator : discriminator for images or videos (general function for both cases)
    # sample_true : Actually is sample_real_image_batch or sample_real_video_batch for sampling real batchsize
    # sample_fake : Actually is sample_images or sample_videos (file models.py) for sampling fake batchsize of videos or images (generated by generator)
    # opt : Adam optimizer for network needed to reduce loss
    # batch_size : size of a batch (batchsize of images differs batchsize of videos)
    def train_discriminator(self, discriminator, sample_true, sample_fake, opt, batch_size, use_categories):
        # Initialize gradients to zeros
        opt.zero_grad()

        # Get real batchsize of images or batchsize of videos (each video in format of many sequent frames)
        real_batch = sample_true()
        batch = Variable(real_batch['images'], requires_grad=False)

        # util.show_batch(batch.data)

        # Get fake batchsize of images or videos
        fake_batch, generated_views, generated_objects, generated_categories = sample_fake(batch_size)

        # Discriminator predicts on real batchsize of images or videos
        real_labels_predict, real_views_predict, real_objects_predict, real_categories_predict = discriminator(batch)

        # Discriminator predicts on fake batchsize of images of videos
        fake_labels_predict, fake_views_predict, fake_objects_predict, fake_categories_predict = discriminator(fake_batch.detach())

        # Ground-truth of real/fake label on real dataset ==> [1,1,... 1]
        ones = self.ones_like(real_labels_predict)
        # Ground-truth of real/fake label on fake dataset ==> [0,0,... 0]
        zeros = self.zeros_like(fake_labels_predict)

        # Loss of Discriminator : Wrong prediction on both real/fake datasets
        l_discriminator = self.gan_criterion(real_labels_predict, ones) + \
                          self.gan_criterion(fake_labels_predict, zeros)

        # Get ground-truth of view label of real videos batch
        views_gt = Variable(torch.squeeze(real_batch['views'].long()), requires_grad=False)
        l_discriminator += self.category_criterion(real_views_predict.squeeze(), views_gt)

        # Get ground-truth of object label of real videos batch
        objects_gt = Variable(torch.squeeze(real_batch['objects'].long()), requires_grad=False)
        l_discriminator += self.category_criterion(real_objects_predict.squeeze(), objects_gt)

        # Incase of videos : Add the loss of wrongly predicting catetories on real dataset
        if use_categories:
            # Get ground-truth of category label of real videos batch
            categories_gt = Variable(torch.squeeze(real_batch['categories'].long()), requires_grad=False)
            l_discriminator += self.category_criterion(real_categories_predict.squeeze(), categories_gt)

        l_discriminator.backward() # calculate gradients in network
        opt.step() # update the learning parameters

        return l_discriminator # return the loss of discriminator before update


    # Calculate loss of generator and update learning parameter using Adam optimizer
    # image_discriminator, video_discriminator : discriminator of image and video
    # sample_fake_images : generate fake batch of images (by generator)
    # sample_fake_videos : generate fake batch of videos (by generator)
    # opt : Optimizer of generator network
    def train_generator(self,
                        image_discriminator, video_discriminator,
                        sample_fake_images, sample_fake_videos,
                        opt):
        # Initialize gradient to zeros
        opt.zero_grad()

        # Train on images
        # Generate fake batch of images
        fake_batch, generated_views, generated_objects, _= sample_fake_images(self.image_batch_size)
        # Discriminator predicts on fake batch of images
        fake_labels_predict, fake_views_predict, fake_objects_predict, _ = image_discriminator(fake_batch)
        # Ground truth of real/fake label ==> Try to fool discriminator ==> [1,1,...1]
        all_ones = self.ones_like(fake_labels_predict)
        # Loss of generator in trying to fool discriminator on real-fake label
        l_generator = self.gan_criterion(fake_labels_predict, all_ones)
        # Loss of generator in trying to fool discriminator on view label
        l_generator += self.category_criterion(fake_views_predict.squeeze(), generated_views)
        # Loss of generator in trying to fool discriminator on object label
        l_generator += self.category_criterion(fake_objects_predict.squeeze(), generated_objects)


        # Train on videos
        # Generate fake batch of videos
        fake_batch, generated_views, generated_objects, generated_categories = sample_fake_videos(self.video_batch_size)
        # Discriminator predicts on fake batch of videos
        fake_labels_predict, fake_views_predict, fake_objects_predict, fake_categories_predict = video_discriminator(fake_batch)
        # Ground truth of real/fake label ==> Try to fool discriminator ==> [1,1,...1]
        all_ones = self.ones_like(fake_labels_predict)
        # Loss of generator in trying to fool discriminator on real-fake label
        l_generator += self.gan_criterion(fake_labels_predict, all_ones)
        # Loss of generator in trying to fool discriminator on view label
        l_generator += self.category_criterion(fake_views_predict.squeeze(), generated_views)
        # Loss of generator in trying to fool discriminator on object label
        l_generator += self.category_criterion(fake_objects_predict.squeeze(), generated_objects)

        # Loss of generator in trying to fool discriminator in terms of categories
        if self.use_infogan:
            # generated_categories is actually ground-truth that generator expect to generate
            # fake_categorical is the function of D(G(x)). This function with unchangeable parameters of D and changeable parameters of G
            # ==> Update the parameters of G to make D predict category of "fake" videos is generated_categories
            l_generator += self.category_criterion(fake_categories_predict.squeeze(), generated_categories)

        l_generator.backward() # Calcualate gradient of generator network
        opt.step() # Update the parameters of generator
        return l_generator



    # Function to train all 3 network : generator, image_discriminator, video_discriminator
    def train(self, generator, image_discriminator, video_discriminator):

        # Generate fake batch of images
        def sample_fake_image_batch(batch_size):
            return generator.sample_images(batch_size)

        # Generate fake batch of videos
        def sample_fake_video_batch(batch_size):
            return generator.sample_videos(batch_size)

        # Save checkpoint to file
        def save_checkpoint(checkpoint, filename_path):
            torch.save(checkpoint, filename_path)


        # Push all networks to gpu to boost computation speed
        if self.use_cuda:
            generator.cuda()
            image_discriminator.cuda()
            video_discriminator.cuda()

        batch_num = 0
        generator_loss = []
        image_discriminator_loss = []
        video_discriminator_loss = []

        # Create optimizers for each network
        opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        opt_image_discriminator = optim.Adam(image_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)
        opt_video_discriminator = optim.Adam(video_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)


        # If it's in resume mode
        if self.resume == True and os.path.isfile(self.checkpoint_path):
            print("=> Loading checkpoint '{}'".format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            batch_num = (checkpoint['batch_num'] + 1) # cause it's the new iteration
            generator_loss = checkpoint['generator_loss']
            image_discriminator_loss = checkpoint['image_discriminator_loss']
            video_discriminator_loss = checkpoint['video_discriminator_loss']
            generator.load_state_dict(checkpoint['generator'])
            image_discriminator.load_state_dict(checkpoint['image_discriminator'])
            video_discriminator.load_state_dict(checkpoint['video_discriminator'])
            opt_generator.load_state_dict(checkpoint['opt_generator'])
            opt_image_discriminator.load_state_dict(checkpoint['opt_image_discriminator'])
            opt_video_discriminator.load_state_dict(checkpoint['opt_video_discriminator'])

        generator_loss.append(0.0)
        image_discriminator_loss.append(0.0)
        video_discriminator_loss.append(0.0)

        # The time to start training
        start_time = time.time()

        while True:

            print("Batch Num = %d, Epoch Num = %d"
                  % (batch_num, (batch_num // self.log_interval)))

            # Set the networks to training mode ==> Because Dropout acts differently in training mode and evaluation mode
            generator.train()
            image_discriminator.train()
            video_discriminator.train()

            # Initialize gradients of all networks to zeros
            opt_generator.zero_grad()
            opt_image_discriminator.zero_grad()
            opt_video_discriminator.zero_grad()

            print("\n\n Train image_discriminator")
            # Train image discriminator
            l_image_dis = self.train_discriminator(image_discriminator, self.sample_real_image_batch,
                                                   sample_fake_image_batch, opt_image_discriminator,
                                                   self.image_batch_size, use_categories=False)
            print("l_image_dis = ", l_image_dis)


            print("\n\n Train video_discriminator")
            # Train video discriminator
            l_video_dis = self.train_discriminator(video_discriminator, self.sample_real_video_batch,
                                                   sample_fake_video_batch, opt_video_discriminator,
                                                   self.video_batch_size, use_categories=self.use_categories)
            print("l_video_dis = ", l_video_dis)

            print("\n\n Train generator")
            # Train generator
            l_gen = self.train_generator(image_discriminator, video_discriminator,
                                         sample_fake_image_batch, sample_fake_video_batch,
                                         opt_generator)
            print("l_gen = ", l_gen)

            generator_loss[-1] += l_gen.data[0]
            image_discriminator_loss[-1] += l_image_dis.data[0]
            video_discriminator_loss[-1] += l_video_dis.data[0]

            # At the end of each epoch
            if batch_num % self.log_interval == (self.log_interval - 1):

                # Average Training Loss
                generator_loss[-1] = generator_loss[-1] / self.log_interval
                image_discriminator_loss[-1] = image_discriminator_loss[-1] / self.log_interval
                video_discriminator_loss[-1] = video_discriminator_loss[-1] / self.log_interval

                # Running Time Per Epoch
                time_per_epoch = time.time() - start_time

                checkpoint = {
                    'batch_num': batch_num,
                    'log_interval' : self.log_interval,
                    'epoch_num': (batch_num // self.log_interval),
                    'generator_loss' : generator_loss,
                    'image_discriminator_loss' : image_discriminator_loss,
                    'video_discriminator_loss' : video_discriminator_loss,
                    'generator': generator.state_dict(),
                    'image_discriminator': image_discriminator.state_dict(),
                    'video_discriminator': video_discriminator.state_dict(),
                    'opt_generator': opt_generator.state_dict(),
                    'opt_image_discriminator' : opt_image_discriminator.state_dict(),
                    'opt_video_discriminator' : opt_video_discriminator.state_dict(),
                    'time_per_epoch' : time_per_epoch
                }


                # Initialize time at the end of epoch
                start_time = time.time()

                # checkpoint_batch_num_epoch_num.pth.tar
                filename_path = os.path.join(self.log_folder, "checkpoint_%06d_%06d.pth.tar" % (
                batch_num, (batch_num // self.log_interval)))
                save_checkpoint(checkpoint, filename_path=filename_path)

                # Add new element to used to calculate loss of next epoch
                if (self.train_batches - batch_num) > self.log_interval :
                    generator_loss.append(0.0)
                    image_discriminator_loss.append(0.0)
                    video_discriminator_loss.append(0.0)

                print("Batch Num = %d, Epoch Num = %d, time_per_epoch = %f"
                      % (batch_num, (batch_num // self.log_interval), time_per_epoch))

                # If it is the last epoch ==> Exit program
                if (self.train_batches - batch_num) <= self.log_interval :
                    break

            # Increase batch_num after each iteration
            batch_num += 1










    # # Function to train all 3 network : generator, image_discriminator, video_discriminator
    # def train(self, generator, image_discriminator, video_discriminator):
    #     # Push all networks to gpu to boost computation speed
    #     if self.use_cuda:
    #         generator.cuda()
    #         image_discriminator.cuda()
    #         video_discriminator.cuda()
    #
    #     # Logger to write log file
    #     logger = Logger(self.log_folder)
    #
    #     # Create optimizers for each network
    #     opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
    #     opt_image_discriminator = optim.Adam(image_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
    #                                          weight_decay=0.00001)
    #     opt_video_discriminator = optim.Adam(video_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
    #                                          weight_decay=0.00001)
    #
    #     # Training loop
    #     # Generate fake batch of images
    #     def sample_fake_image_batch(batch_size):
    #         return generator.sample_images(batch_size)
    #
    #     # Generate fake batch of videos
    #     def sample_fake_video_batch(batch_size):
    #         return generator.sample_videos(batch_size)
    #
    #     # To write log ???
    #     def init_logs():
    #         return {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}
    #
    #     # Count total batches of training process
    #     batch_num = 0
    #
    #     # Create a dictionary object to save loss of generator, image_discriminator, and video_discriminator
    #     logs = init_logs()
    #
    #     # The time to start training
    #     start_time = time.time()
    #
    #     while True:
    #         # Set the networks to training mode ==> Because Dropout acts differently in training mode and evaluation mode
    #         generator.train()
    #         image_discriminator.train()
    #         video_discriminator.train()
    #
    #         # Initialize gradients of all networks to zeros
    #         opt_generator.zero_grad()
    #         opt_image_discriminator.zero_grad() # Why author don't use this command before ??? is it wrong?
    #         opt_video_discriminator.zero_grad()
    #
    #         # Train image discriminator
    #         l_image_dis = self.train_discriminator(image_discriminator, self.sample_real_image_batch,
    #                                                sample_fake_image_batch, opt_image_discriminator,
    #                                                self.image_batch_size, use_categories=False)
    #
    #         # Train video discriminator
    #         l_video_dis = self.train_discriminator(video_discriminator, self.sample_real_video_batch,
    #                                                sample_fake_video_batch, opt_video_discriminator,
    #                                                self.video_batch_size, use_categories=self.use_categories)
    #
    #         # Train generator
    #         l_gen = self.train_generator(image_discriminator, video_discriminator,
    #                                      sample_fake_image_batch, sample_fake_video_batch,
    #                                      opt_generator)
    #
    #
    #         # Accumulate loss of networks G, DI, DV after each batch optimization
    #         logs['l_gen'] += l_gen.data[0]
    #         logs['l_image_dis'] += l_image_dis.data[0]
    #         logs['l_video_dis'] += l_video_dis.data[0]
    #
    #         # Trace the number of batches that went over
    #         batch_num += 1
    #
    #         # Prints results to log file after each log_interval batches
    #         if batch_num % self.log_interval == 0:
    #             # Print average loss on one batch + The time long to this training point onto screen
    #             log_string = "Batch %d" % batch_num
    #             for k, v in logs.iteritems():
    #                 log_string += " [%s] %5.3f" % (k, v / self.log_interval)
    #
    #             log_string += ". Took %5.2f" % (time.time() - start_time)
    #             print log_string
    #
    #             # Save logs to file
    #             for tag, value in logs.items():
    #                 logger.scalar_summary(tag, value / self.log_interval, batch_num)
    #
    #             # Initialize logs dictionary with accumulated losses be zeros
    #             logs = init_logs()
    #             # Set new start time at this point ==> Dont agree with this trategy
    #             start_time = time.time()
    #
    #             # Set network in evaluation mode cause Dropout act differently in training mode and evaluation mode
    #             generator.eval()
    #
    #             images, _ = sample_fake_image_batch(self.image_batch_size)
    #             logger.image_summary("Images", images_to_numpy(images), batch_num)
    #
    #             videos, _ = sample_fake_video_batch(self.video_batch_size)
    #             logger.video_summary("Videos", videos_to_numpy(videos), batch_num)
    #
    #             torch.save(generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))
    #
    #         if batch_num >= self.train_batches:
    #             torch.save(generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))
    #             break
