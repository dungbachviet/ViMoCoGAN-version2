#!/usr/bin/python
# -*- coding: latin-1 -*-
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

from __future__ import division, print_function, unicode_literals
import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import PIL
from train import args



# Create VideoFolderDataset containing total data
# folder : path to root folder to contain data
# cache : path to object file saving important informations (image paths, lengths)
# min_len : minimum frames to have in one video
class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache, min_len=16):
        dataset = ImageFolder(folder) # Create dataset object in format of ImageFolder (root directory only contains a direct children layer of action categories)
        self.total_frames = 0 # Total frames in every videos in dataset
        self.lengths = [] # Save number of frames of each video in dataset
        self.images = [] # Save path of every videos in format of "long images" and catetory of that video (an image is a long sequence of frames concatenated)


        # Check the path to cache whether exists or not
        if cache is not None and os.path.exists(cache):
            with open(cache, 'r') as f:
                self.images, self.lengths = pickle.load(f) # Load list of image paths and lengths
        else:
            # If not existing, build list of image paths and lengths again
            for idx, (im, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx] # (image_path, class_index_of_that_image)
                shorter, longer = min(im.width, im.height), max(im.width, im.height)# Get longer side and shorter side of "long" image
                length = longer // shorter # Get number of frames of video (means 'long' image)
                if length >= min_len: # Only get videos having more than min_len frames
                    print("img_path = ", img_path)
                    # /src/action_view_data_giang_long/0001_0004/0001_0.png
                    token = img_path.split('/')
                    file_name = token[-1]  # 0001_0.png
                    object_id = int(file_name[-5:-4])  # 0

                    # view_id, action_id = divmod(categ, 12)
                    view_id, action_id = divmod(categ, int(args["--dim_z_category"]))
                    self.images.append((img_path, view_id, action_id, object_id))  # Save path of video ('long' image) and category
                    self.lengths.append(length)  # Save number of frames of the video


            # If indicating the path to save cache ==> Save self.images and lengths to use at later times
            if cache is not None:
                with open(cache, 'w') as f:
                    pickle.dump((self.images, self.lengths), f) # save object using pickle

        # Save accumulated sequence of number of frames in videos
        # Ex : [5,10,8,5] ==> [0,5,15,23,28] ==> total frames = 28, if frame_idx=17 --> lies at 3th frame of 2th video
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of videos {}".format(len(self.lengths)))
        print("Total number of frames {}".format(np.sum(self.lengths)))

    # Override method to get one instance in dataset
    # item is an index of domain : 0-->(number of videos - 1)
    def __getitem__(self, item):
        path, view_id, action_id, object_id = self.images[item] # path of "long" image and category of the video
        im = PIL.Image.open(path) # PIL format of "long" image
        return im, view_id, action_id, object_id # PIL of video and a scalar (indicates category index of the video)

    # Override method to count length of VideoFolderDataset = number of videos ("long" images)
    def __len__(self):
        return len(self.images)


# Dataset to get one image
# dataset : actually is VideoFolderDataset (above)
# transform : to tranform for taken image
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transforms = transform if transform is not None else lambda x: x


    # Get one image randomly from dataset
    # item : is index of image, with value domain : 0--> (total frames of dataset - 1)
    def __getitem__(self, item):
        if item != 0:
            # From accumulated sequence [0,5,10,15,23] ==> From item (frame idex) find video index
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            # After finding video_id ==> Indicate the frame index in that video
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else: # if item=0 ==> video_id=0, frame_num=0
            video_id = 0
            frame_num = 0

        # Get "long" PIL image and scalar indicating category
        video, view_id, action_id, object_id = self.dataset[video_id]
        video = np.array(video)  # convert PIL "long" image to numpy : (height, width, channel)

        # Boolean to know if "long" numpy spreads to horizontal direction
        horizontal = video.shape[1] > video.shape[0]

        # Determine the location to cut one frame from a certain video ("long" image numpy)
        if horizontal:
            i_from, i_to = video.shape[0] * frame_num, video.shape[0] * (frame_num + 1)
            frame = video[:, i_from: i_to, ::] # ký hiệu :: chỉ là lấy ra toàn bộ các phần tử của chiều thứ 3 thôi !!!
        else:
            i_from, i_to = video.shape[1] * frame_num, video.shape[1] * (frame_num + 1)
            frame = video[i_from: i_to, :, ::]

        # ??? Never happen
        if frame.shape[0] == 0:
            print("video {}. From {} to {}. num {}".format(video.shape, i_from, i_to, item))

        # Return the taken image frame and category (not necessary)
        # If dataload gets batchsize of 4 images ==> "images" will contain 4 taken frames
        # "images": (batchsize, height, width, channel) ==> transform : (batchsize, channel,height, width) with value domain -1 --> 1 (check out image_transform in train.py)
        # "categories" ==> vector of size (batchsize)
        return {"images": self.transforms(frame), "views": view_id, "categories": action_id, "objects":object_id}

    # Override methods to indicate length of dataset = Total number of image frames in dataset
    def __len__(self):
        return self.dataset.cumsum[-1]


# Dataset to get one video
# dataset : actually is VideoFolderDataset (above)
# video_length : fixed length of frames in videos
# every_nth : Get every n'th frames to be equal with fixed length of frames (video_length) if video's too long
# transform : to tranform for taken video
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth
        self.transforms = transform if transform is not None else lambda x: x

    # Get one video randomly in dataset
    def __getitem__(self, item):
        video, view_id, action_id, object_id = self.dataset[item] # Get one video ("long" PIL image) and its category
        video = np.array(video) # Convert from PIL to array

        # Check the "long" array to see what side it spreads to
        horizontal = video.shape[1] > video.shape[0]
        # Get number of frames of the given video
        shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
        video_len = longer // shorter

        # If video's longer than fixed video length ==> Get frames at every n'th frames
        if video_len >= self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            raise Exception("Length is too short id - {}, len - {}").format(self.dataset[item], video_len)

        # Slit "long" numpy into video_len numpy arrays represents corresponding image frames
        # (video_len, height, width, channel)
        frames = np.split(video, video_len, axis=1 if horizontal else 0)

        # Only get video_length frames in total of video_len frames of video
        # Size : (video_length, height, width, channel)
        selected = np.array([frames[s_id] for s_id in subsequence_idx])

        # Dataloader get batchsize of videos
        # "images" ==> (batchsize, video_length, height, width, channel)
        # "images" after transformed ==> (batchsize, channel, depth=video_lenght, height, width)
        # "categories" ==> vector (batchsize), each element indicates category of the video in batchsize
        return {"images": self.transforms(selected), "views": view_id, "categories": action_id, "objects": object_id}

    # Indicate the size of VideoDataset = number of videos in dataset
    def __len__(self):
        return len(self.dataset)


# Don't have any need to use ImageSampler and VideoSampler ('ll find more later)
class ImageSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transforms = transform


    def __getitem__(self, index):
        result = {}
        for k in self.dataset.keys:
            result[k] = np.take(self.dataset.get_data()[k], index, axis=0) # Lấy dữ liệu có các chỉ mục là index trong dữ liệu self.dataset.get_data()[k]

        if self.transforms is not None:
            for k, transform in self.transforms.iteritems():
                result[k] = transform(result[k])

        return result


    def __len__(self):
        return self.dataset.get_data()[self.dataset.keys[0]].shape[0]


class VideoSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.unique_ids = np.unique(self.dataset.get_data()['video_ids'])
        self.every_nth = every_nth
        self.transforms = transform

    def __getitem__(self, item):
        result = {}
        ids = self.dataset.get_data()['video_ids'] == self.unique_ids[item]
        ids = np.squeeze(np.squeeze(np.argwhere(ids)))
        for k in self.dataset.keys:
            result[k] = np.take(self.dataset.get_data()[k], ids, axis=0)

        subsequence_idx = None
        print(result[k].shape[0])

        # videos can be of various length, we randomly sample sub-sequences
        if result[k].shape[0] > self.video_length:
            needed = self.every_nth * (self.video_length - 1)
            gap = result[k].shape[0] - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif result[k].shape[0] == self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            print("Length is too short id - {}, len - {}".format(self.unique_ids[item], result[k].shape[0]))

        if subsequence_idx:
            for k in self.dataset.keys:
                result[k] = np.take(result[k], subsequence_idx, axis=0)
        else:
            print(result[self.dataset.keys[0]].shape)

        if self.transforms is not None:
            for k, transform in self.transforms.iteritems():
                result[k] = transform(result[k])

        return result

    def __len__(self):
        return len(self.unique_ids)