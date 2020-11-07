import os
import crnn
import numpy as np
import cv2
from tqdm import tqdm
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


mtcnn = MTCNN(select_largest=True, post_process=True, device='cuda:0')
m = InceptionResnetV1(pretrained='vggface2')

def crop(video_path, filename):

    # Load a video
    v_cap = cv2.VideoCapture(video_path + filename)
    t = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    
    n = 10
    frame_count = 60
    skip = frame_count // n
    
    t_s = (t-frame_count) // 2
    t_e = t_s + frame_count

    if t < frame_count:
        t_s = 0
        t_e = t
        skip = t // n
        frame_count = t

    for i in range(t):

        # Load frame
        success = v_cap.grab()
        if i > t_e:
            break
        if i >= t_s and i < t_e:
            if i % skip == 0:
                success, frame = v_cap.retrieve()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                continue
        if not success:
            continue

    # Detect faces in batch
    faces = mtcnn(frames)
    X = torch.stack(faces)
    torch.save(X, 'torch_video_10/' +filename[:5]+'.pt')
    
data_dir = "../emotion/qia2020/train/"
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".mp4"): 
        continue 
    crop(data_dir, filename)