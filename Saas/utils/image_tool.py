
import cv2
import sys
import numpy as np
from PIL import Image

def concat_images(img1, img2, img3):
    # 获取图片的高、宽、通道数
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    h3, w3, c3 = img3.shape

    # 计算拼接图片的高、宽、通道数
    w = w1 + w2 + w3
    h = max(h1, h2, h3)
    c = c1
    # 创建空的拼接图片
    img = np.zeros((h, w, c), np.uint8)

    # 在拼接图片上按位置填充三张图片的像素
    img[:h1, 0:w1, :] = img1
    img[:h2, w1:w1 + w2, :] = img2
    img[:h3, w1 + w2:w, :] = img3

    return img
