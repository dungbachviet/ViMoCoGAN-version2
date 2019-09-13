
import os, os.path
import shutil
import numpy as np
import random
import re
from PIL import Image


# Convert number to string of format "xxxx" (0001, 0002, ... 0100, ... 9999)
def numberToString(number) :
    if number < 10 :
        return "000" + str(number)
    elif number < 100 :
        return "00" + str(number)
    elif number < 1000 :
        return "0" + str(number)
    else :
        return str(number)


def restructureFolders(ROOT, kinect_set, object_set, action_set) :
    # Rename frames in video in increasing order with format "xxx.png" (001.png, 002.png,... 999.png)
    # Remove unnecessary directories
    for kinect in kinect_set:
        if os.path.isdir(os.path.join(ROOT, "%s" % (kinect))):
            for object in object_set:
                if os.path.isdir(os.path.join(ROOT, "%s/%s" % (kinect, object))):
                    for action in action_set:
                        if os.path.isdir(os.path.join(ROOT, "%s/%s/%s" % (kinect, object, action))):
                            listing_dir = os.path.join(ROOT, "%s/%s/%s" % (kinect, object, action))
                            # ["1.1", "1.2","1.3",...]
                            action_times_set = sorted([name for name in os.listdir(listing_dir) if
                                                    os.path.isdir(os.path.join(listing_dir, name))])
                            for action_times in action_times_set:
                                # 1.1_RGB
                                action_times_type = action_times + "_RGB"
                                DIR = os.path.join(ROOT, '%s/%s/%s/%s/%s' % (kinect, object, action, action_times, action_times_type))
                                if os.path.isdir(DIR):
                                    list_file_name = sorted(
                                        [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

                                    # If directory is empty ==> Remove the empty directory
                                    if len(list_file_name) == 0 :
                                        shutil.rmtree(
                                            os.path.join(ROOT, "%s/%s/%s/%s" % (kinect, object, action, action_times)))
                                        continue

                                    # Display name of directory
                                    print(DIR)  # './Full_segmented/Segment_Kinect_1/Binh/2/2.4/2.4_RGB'
                                    # Display files in directory
                                    print sorted(
                                        [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

                                    count = 0
                                    # Rename of files in increasing order
                                    for file_name in sorted([name for name in os.listdir(DIR) if
                                                             os.path.isfile(os.path.join(DIR, name))]):
                                        count += 1
                                        os.rename(os.path.join(DIR, file_name),
                                                  os.path.join(DIR, "%s.png" % (numberToString(count))))

                                    # Move the directory to other location
                                    os.rename(DIR, os.path.join(ROOT, "%s/%s/%s/%s" % (kinect, object, action, action_times_type)))
                                    shutil.rmtree(os.path.join(ROOT, "%s/%s/%s/%s" % (kinect, object, action, action_times)))

    # Create 60 directories = 5 views x 12 actions
    for kinect in range(1, 6):
        for (action) in range(1, 13):
            new_directory = os.path.join(ROOT, "%s_%s" % (numberToString(kinect), numberToString(action)))
            if os.path.isdir(new_directory) == False:
                try:
                    os.mkdir(new_directory)
                except OSError:
                    print ("Creation of the directory %s failed" % (new_directory))

    # Put videos having same view and action into one common directory
    for kinect in range(1, 6):
        for action in range(1, 13):
            count_videos = 0
            dest_folder = os.path.join(ROOT, "%s_%s" % (numberToString(kinect), numberToString(action)))
            for object in object_set:
                listing_dir = os.path.join(ROOT, "%s/%s/%s" % ("Segment_Kinect_" + str(kinect), object, str(action)))
                if os.path.isdir(listing_dir):
                    for dir_name in sorted([name for name in os.listdir(listing_dir)
                                            if os.path.isdir(os.path.join(listing_dir, name))]):
                        count_videos += 1
                        os.rename(os.path.join(listing_dir, dir_name),
                                  os.path.join(dest_folder, numberToString(count_videos)))

    # Remove the unnecessary remaining directories
    for kinect in kinect_set:
        removed_directory = os.path.join(ROOT, kinect)
        if os.path.isdir(removed_directory) == True:
            shutil.rmtree(removed_directory)


def createStatisticFile(ROOT, file_name):
    statistic_file_path = os.path.join(ROOT, file_name)
    with open(statistic_file_path, 'w') as f_statistic :
        for kinect in range(1,6):
            for action in range(1,13):
                directory = os.path.join(ROOT, "%s_%s" % (numberToString(kinect), numberToString(action)))
                if os.path.isdir(directory) :
                    list_video_name = sorted([name for name in os.listdir(directory)
                            if os.path.isdir(os.path.join(directory, name))])
                    print("%s %s %d" % (kinect, action, len(list_video_name)))
                    f_statistic.write("%s %s %d\n" % (kinect, action, len(list_video_name)))


def getStatisticInfo(ROOT, file_name) :
    statistic_file_path = os.path.join(ROOT, file_name)
    view_action_groups = []
    num_videos_per_lalels = []
    num_frames_per_videos = []
    with open(statistic_file_path, 'r') as f_statistic :
        for video_list in f_statistic.readlines() :
            line_to_list = map(int, video_list.strip().split(" ")) # [view, action, num_videos]
            view_action_groups.append(line_to_list)
            num_videos_per_lalels.append(line_to_list[2])

            video_dir = os.path.join(ROOT, "%s_%s" % (numberToString(line_to_list[0]), numberToString(line_to_list[1])))
            for video_index in range(line_to_list[2]):
                frame_dir = os.path.join(video_dir, "%s" % numberToString(video_index+1))
                num_frames_per_videos.append(len(os.listdir(frame_dir)))

    return view_action_groups, num_videos_per_lalels, num_frames_per_videos

# Statistic : Choose proper number of frames in video
def average_mean_frames(num_frames_per_videos):
    sorted_list = sorted(num_frames_per_videos)
    print("\nSorted List : ", sorted_list)
    print("Average of frames : ", np.average(sorted_list))
    print("Median of frames : ", np.median(sorted_list))
    print("Total videos : ", len(sorted_list))
    print("Total frames : ", sum(sorted_list))


def split_train_valid_test(ROOT, train_root, valid_root, test_root, train_rate, valid_rate, test_rate) :
    if not os.path.exists(train_root): os.mkdir(train_root)
    if not os.path.exists(valid_root): os.mkdir(valid_root)
    if not os.path.exists(test_root): os.mkdir(test_root)
    for view in range(1, 6):
        for action in range(1, 13):
            if not os.path.exists(os.path.join(train_root, "%s_%s" % (numberToString(view), numberToString(action)))):
                os.mkdir(os.path.join(train_root, "%s_%s" % (numberToString(view), numberToString(action))))
            if not os.path.exists(os.path.join(valid_root, "%s_%s" % (numberToString(view), numberToString(action)))):
                os.mkdir(os.path.join(valid_root, "%s_%s" % (numberToString(view), numberToString(action))))
            if not os.path.exists(os.path.join(test_root, "%s_%s" % (numberToString(view), numberToString(action)))):
                os.mkdir(os.path.join(test_root, "%s_%s" % (numberToString(view), numberToString(action))))

    for view in range(1, 6):
        for action in range(1, 13):
            directory = os.path.join(ROOT, "%s_%s" % (numberToString(view), numberToString(action)))
            train_directory = os.path.join(train_root, "%s_%s" % (numberToString(view), numberToString(action)))
            valid_directory = os.path.join(valid_root, "%s_%s" % (numberToString(view), numberToString(action)))
            test_directory = os.path.join(test_root, "%s_%s" % (numberToString(view), numberToString(action)))

            video_name_list = sorted(
                [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])
            num_videos = len(video_name_list)
            num_train_videos = int(train_rate * num_videos)
            num_test_videos = int(test_rate * num_videos)
            num_valid_videos = num_videos - num_train_videos - num_test_videos

            video_indices = range(num_videos)
            random.shuffle(video_indices)
            train_id = 0
            valid_id = 0
            test_id = 0
            for i in range(num_videos):
                copy_directory = os.path.join(directory, "%s" % (numberToString(video_indices[i] + 1)))
                if i < num_train_videos:
                    train_id += 1
                    dist_directory = os.path.join(train_directory, "%s" % (numberToString(train_id)))
                    shutil.copytree(copy_directory, dist_directory)
                elif i < num_train_videos + num_valid_videos:
                    valid_id += 1
                    dist_directory = os.path.join(valid_directory, "%s" % (numberToString(valid_id)))
                    shutil.copytree(copy_directory, dist_directory)
                else:
                    test_id += 1
                    dist_directory = os.path.join(test_directory, "%s" % (numberToString(test_id)))
                    shutil.copytree(copy_directory, dist_directory)

#
# # Run these code to structure the directory to standard format
# kinect_set = ["Segment_Kinect_1", "Segment_Kinect_2", "Segment_Kinect_3", "Segment_Kinect_4", "Segment_Kinect_5"]
# object_set = ["Binh", "Giang", "Hoang", "Hung", "Tan", "Thuan"]
# action_set = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
# ROOT = './Full_segmented'
# file_name = "statistic_file.txt"
# train_rate, valid_rate, test_rate = 0.7, 0.15, 0.15
# train_root = "./train_data"
# valid_root = "./valid_data"
# test_root = "./test_data"
#
# # Restructe Folders :
# restructureFolders(ROOT, kinect_set, object_set, action_set)
#
# # Statistics in ROOT folder (all datasets)
# print("\n\nStatistics in ROOT folder (all datasets)")
# createStatisticFile(ROOT, file_name)
# view_action_groups, num_videos_per_lalels, num_frames_per_videos = getStatisticInfo(ROOT, file_name)
# average_mean_frames(num_frames_per_videos)
#
# num_frames_per_videos = ([0] + num_frames_per_videos)
# num_frames_per_videos = np.cumsum(num_frames_per_videos)
#
# num_videos_per_lalels = ([0] + num_videos_per_lalels)
# num_videos_per_lalels = np.cumsum(num_videos_per_lalels)
#
# for i in range(len(num_frames_per_videos)-1):
#     if num_frames_per_videos[i] == num_frames_per_videos[i+1] :
#         print("***Warning ROOT: index i = ", i)
#         print(num_frames_per_videos[i])
#         print(num_frames_per_videos[i+1])
#         label_id = np.searchsorted(num_videos_per_lalels, i + 1) - 1
#         # video at x'th position in label label_id
#         video_id_in_label = i - num_videos_per_lalels[label_id]
#         view_id, action_id = divmod(label_id, 12)
#         print(view_id+1, action_id+1, video_id_in_label+1)
#
#
#
# # Split into 3 dataset for training, validation, testing
# split_train_valid_test(ROOT, train_root, valid_root, test_root, train_rate, valid_rate, test_rate)
#
# # Statistics in train_data folder (train data)
# print("\n\nStatistics in train_data folder (train data)")
# createStatisticFile(train_root, file_name)
# view_action_groups, num_videos_per_lalels, num_frames_per_videos = getStatisticInfo(train_root, file_name)
# average_mean_frames(num_frames_per_videos)
#
#
#
# num_frames_per_videos = ([0] + num_frames_per_videos)
# test = np.cumsum(num_frames_per_videos)
# print("test = ", test)
# for i in range(len(test)-1):
#     if test[i] == test[i+1] :
#         print("***Warning train: index i = ", i)
#         print(test[i])
#         print(test[i+1])
#
#
# # Statistics in valid_data folder (validation data)
# print("\n\nStatistics in valid_data folder (validation data)")
# createStatisticFile(valid_root, file_name)
# view_action_groups, num_videos_per_lalels, num_frames_per_videos = getStatisticInfo(valid_root, file_name)
# average_mean_frames(num_frames_per_videos)
#
#
# num_frames_per_videos = ([0] + num_frames_per_videos)
# test = np.cumsum(num_frames_per_videos)
# print("test = ", test)
# for i in range(len(test)-1):
#     if test[i] == test[i+1] :
#         print("***Warning valid: index i = ", i)
#         print(test[i])
#         print(test[i+1])
#
#
#
# # Statistics in test_data folder (test data)
# print("\n\nStatistics in test_data folder (test data)")
# createStatisticFile(test_root, file_name)
# view_action_groups, num_videos_per_lalels, num_frames_per_videos = getStatisticInfo(test_root, file_name)
# average_mean_frames(num_frames_per_videos)
#
#
# num_frames_per_videos = ([0] + num_frames_per_videos)
# test = np.cumsum(num_frames_per_videos)
# print("test = ", test)
# for i in range(len(test)-1):
#     if test[i] == test[i+1] :
#         print("***Warning test: index i = ", i)
#         print(test[i])
#         print(test[i+1])





# ROOT = "./overfit_data"
# file_name = "statistic_file.txt"
# def creat_overfit_data(ROOT, file_name) :
#     statistic_file_path = os.path.join(ROOT, file_name)
#     with open(statistic_file_path, 'w') as f_statistic :
#         for kinect in range(1,2):
#             for action in range(1,2):
#                 directory = os.path.join(ROOT, "%s_%s" % (numberToString(kinect), numberToString(action)))
#                 if os.path.isdir(directory) :
#                     list_video_name = sorted([name for name in os.listdir(directory)
#                             if os.path.isdir(os.path.join(directory, name))])
#                     print("%s %s %d" % (kinect, action, len(list_video_name)))
#                     f_statistic.write("%s %s %d\n" % (kinect, action, len(list_video_name)))
#
#
#     view_action_groups, num_videos_per_lalels, num_frames_per_videos = getStatisticInfo(ROOT, file_name)
#
#     print(view_action_groups)
#     print(num_videos_per_lalels)
#     print(np.sum(num_frames_per_videos))



# # Convert jpg to png, reconstruct and rename file's names
# root_folder = "./frames"
# count_folder = 0
# for folder_name in sorted([name for name in os.listdir(root_folder)
#                             if os.path.isdir(os.path.join(root_folder, name))]) :
#     folder_path = os.path.join(root_folder, folder_name)
#     count_folder += 1
#     count_file = 0
#
#     for file_name in ([name for name in os.listdir(folder_path)
#                             if os.path.isfile(os.path.join(folder_path, name))]) :
#         print("filename = ", file_name)
#         number = int(re.search(r'[0-9]+', file_name).group())
#
#         os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, numberToString(number) + "_test.jpg"))
#
#     for file_name in sorted([name for name in os.listdir(folder_path)
#                        if os.path.isfile(os.path.join(folder_path, name))]):
#         print("filename = ", file_name)
#         count_file += 1
#         im = Image.open(os.path.join(folder_path, file_name))
#         im.save(os.path.join(folder_path, numberToString(count_file) + ".png"))
#         os.remove(os.path.join(folder_path, file_name))
#
#     if count_file == 0 : print("Folder folder_path's empty")
#     os.rename(os.path.join(root_folder, folder_name), os.path.join(root_folder, numberToString(count_folder)))


ROOT = "./my_hand_data"
file_name = "statistic_file.txt"
def create_overfit_data(ROOT, file_name) :
    statistic_file_path = os.path.join(ROOT, file_name)
    with open(statistic_file_path, 'w') as f_statistic :
        for kinect in range(1,2):
            for action in range(1,2):
                directory = os.path.join(ROOT, "%s_%s" % (numberToString(kinect), numberToString(action)))
                if os.path.isdir(directory) :
                    list_video_name = sorted([name for name in os.listdir(directory)
                            if os.path.isdir(os.path.join(directory, name))])
                    print("%s %s %d" % (kinect, action, len(list_video_name)))
                    f_statistic.write("%s %s %d\n" % (kinect, action, len(list_video_name)))


create_overfit_data(ROOT, file_name)
view_action_groups, num_videos_per_lalels, num_frames_per_videos = getStatisticInfo(ROOT, file_name)

print(view_action_groups)
print(num_videos_per_lalels)
print(np.sum(num_frames_per_videos))
