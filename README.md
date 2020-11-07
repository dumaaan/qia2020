# VATA: Video Attention, Text Attention

## This is the implementation of our final model for the QIA 2020 Challendge on [Kaggle](https://www.kaggle.com/c/qia-hackathon2020/overview).

How to run: 

1. Download the dataset from Kaggle.
2. Use [MTCNN](https://github.com/ipazc/mtcnn) and [OpenCV](https://github.com/opencv/opencv) to crop out the faces from the video.
3. Transform frames into PyTorch tensors.
4. Run the training file: `python fe_train.py`

## Structure
![Model](https://i.ibb.co/6YFzYbL/Screen-Shot-2020-11-07-at-2-07-37-PM.png)
