'''
Using OpenCV takes a mp4 video and produces a number of images.
Requirements
----
You require OpenCV 3.2 to be installed.
Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py
Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

#
# pathIn = "./mica_hand_videos/"
# pathOut = "./data/actions/"
# actions = ["action4", "action5", "action8", "action9"]
#
# try:
#     if not os.path.exists(pathOut):
#         os.makedirs(pathOut)
# except OSError:
#     print ('Error: Creating directory of data')
#
# for action_name in actions :
#     scan_directory = os.path.join(pathIn, action_name)
#     for index, file_name in enumerate(sorted([name for name in os.listdir(scan_directory)
#             if os.path.isfile(os.path.join(scan_directory, name))])):
#
#         video_path = os.path.join(scan_directory, file_name)
#         print(video_path)
#         video_cap = cv2.VideoCapture(video_path)
#         frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
#         milisecond_per_frame = int((1 / frames_per_second) * 1000)
#         print("milisecond_per_frame = ", milisecond_per_frame)
#
#         num_frames = (int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)  # actually having less than 1 frame
#         # xet them truong hop neu num_frames < 16
#         chosen_frames = list(np.linspace(1, num_frames, 16, dtype=np.int))
#         list_images = []
#
#         for frame_id in chosen_frames:
#             video_cap.set(cv2.CAP_PROP_POS_MSEC, (frame_id * milisecond_per_frame))
#             success_state, image = video_cap.read()
#             if (success_state == True):
#                 resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
#                 list_images.append(resized)
#
#         print("num_frames = ", num_frames)
#         print("chosen_frames = ", chosen_frames)
#         print("==> Save long image")
#
#         long_image = np.concatenate(list_images, axis=1)
#         print("long_image.shape = ", long_image.shape)
#         cv2.imwrite(pathOut + "%s/%04d.png" % (action_name, index+1), np.array(long_image))
#



# performers = ["001_Giang", "002_VuHai", "003_NguyenTrongTuyen", "004_TranDucLong", "005_TranThiThuThuy",
#               "006_KhongVanMinh", "007_BuiHaiPhong", "008_NguyenThiThanhNhan", "009_Binh", "010_Tan",
#               "011_Thuan"]

# views = ["0001", "0002", "0003", "0004", "0005"]
# actions = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009",
#           "0010", "0011", "0012"]
# root_path = "/home/dungbachviet/Desktop/mica_hand_videos"
# destinatation_path = "./action_view_data/"


performers = ["001_Giang", "004_TranDucLong"]

views = ["0001", "0002", "0003", "0004", "0005"]
actions = ["0004", "0005", "0008", "0009"]
root_path = "/media/dungbachviet/FC2A9B962A9B4C90/DoAnTotNghiep/mica_hand_videos"
destinatation_path = "./action_view_data_giang_long/"

# Create directories of format : viewId_actionId
for view_id in views:
    for action_id in actions:
        created_path = os.path.join(destinatation_path, "%s_%s" % (view_id, action_id))
        if (not os.path.exists(created_path)):
            os.makedirs(created_path)

# Reconstruct data and move to destination path
for view_id in views:
    for action_id in actions:
        count_video = 0 # count number of videos in the same label (view + action)
        for performer_idx in range(len(performers)):

            performer_id = performers[performer_idx]
            # Check path to /home/dungbachviet/Desktop/mica_hand_videos/001_Giang
            if (not os.path.exists(os.path.join(root_path, "%s" % performer_id))):
                print("Not exist path to %s", os.path.join(root_path, "%s" % performer_id))
                continue

            # Check path to /home/dungbachviet/Desktop/mica_hand_videos/001_Giang/0001
            if (not os.path.exists(os.path.join(root_path, "%s/%s" % (performer_id, view_id)))):
                print("Not exist path to %s", os.path.join(root_path, "%s/%s" % (performer_id, view_id)))
                continue


            # path to view :/home/dungbachviet/Desktop/mica_hand_videos/001_Giang/0001
            scan_directory = os.path.join(root_path, "%s/%s" % (performer_id, view_id))
            for folder_name in sorted([name for name in os.listdir(scan_directory) if os.path.isdir(os.path.join(scan_directory, name))]):
                print("\n\nfolder_name : ", folder_name)
                action = folder_name.strip().split("_") # action_performNum
                print("file_name after splitting : ", action)

                if (int(action[0]) == int(action_id)):
                    count_video += 1
                    # Get out a video to convert to a "long image"
                    video_path = os.path.join(scan_directory, "%s/%s" % (folder_name, "video.avi"))

                    # Check path to /home/dungbachviet/Desktop/mica_hand_videos/001_Giang/0001/1_1/video.avi
                    if (not os.path.exists(os.path.join(scan_directory, "%s/%s" % (folder_name, "video.avi")))):
                        print("Not exist path to %s", os.path.join(scan_directory, "%s/%s" % (folder_name, "video.avi")))
                        continue

                    print("video_path = ", video_path)

                    # object to manage information of video
                    video_cap = cv2.VideoCapture(video_path)
                    frames_per_second = video_cap.get(cv2.CAP_PROP_FPS)
                    print("frames_per_second = ", frames_per_second)

                    milisecond_per_frame = int((1 / frames_per_second) * 1000)
                    print("milisecond_per_frame = ", milisecond_per_frame)

                    num_frames = (int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)  # actually having less than 1 frame
                    print("num_frames = ", num_frames)
                    # xet them truong hop neu num_frames < 16 ==> automatically duplicate one frame multiple times
                    chosen_frames = list(np.linspace(1, num_frames, 16, dtype=np.int))
                    print("chosen_frames = ", chosen_frames)

                    # Save frames split from videos
                    list_images = []
                    for frame_id in chosen_frames:
                        # Refer to the location to get the expected frame
                        # One frame can be got out multiple times (if total frames less than 16)
                        video_cap.set(cv2.CAP_PROP_POS_MSEC, (frame_id * milisecond_per_frame))
                        success_state, image = video_cap.read()

                        # Resize each frame to size of (64,64)
                        if (success_state == True):
                            resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                            list_images.append(resized)

                    print("==> Save long image")
                    # Concatenate all frames into a "long image"
                    long_image = np.concatenate(list_images, axis=1)
                    print("long_image.shape = ", long_image.shape)

                    # path to save the "long image" of a video
                    save_video_to = os.path.join(destinatation_path, "%s_%s/%04d_%d.png" % (view_id, action_id, count_video, performer_idx))
                    cv2.imwrite(save_video_to, np.array(long_image))

