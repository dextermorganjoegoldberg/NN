# -*- coding:utf-8 -*-
"""
Title: one-layer convolution
Author:He Hulingxiao
Date:2022.10.29
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from math import ceil

def conv_per_channel(data, kernel, padding, stride):
    h_k, w_k = kernel.shape
    hr = h_k // 2
    wr = w_k // 2
    h, w = data.shape
    #边界填充后的image map
    padding_data = np.zeros([h+2*padding, w+2*padding], np.float32)
    #保存卷积后图像
    result = np.zeros([ceil((h+2*padding-h_k+1) / stride),ceil((w+2*padding-w_k+1) / stride)], np.float32)
    #将输入图像在非padding区域填充
    padding_data[padding:h + padding, padding:w + padding] = data
    #对每个像素进行遍历
    for i in range(padding, h  + padding, stride):
        for j in range(padding, w + padding, stride):
            # 取出当前像素的h_k x w_k 邻域
            neighbor = padding_data[i-hr: i+hr+1, j-wr: j+wr+1]
            # 计算该点的卷积值
            result[(i-padding)//2][(j-padding)//2] = np.sum(neighbor * kernel)

    return result

def conv_per_kernel(input, kernel, padding, stride):
    h, w, c = input.shape
    h_k, w_k, c_k = kernel.shape
    outputs = np.zeros([ceil((h+2*padding-h_k+1) / stride),ceil((w+2*padding-w_k+1) / stride)])
    assert c_k == c, "The num of channels of kernel and img should be the same"
    # 对每个channel进行遍历，从而对每个channel进行卷积
    for i in range(c):
        f_map = input[:,:,i]
        w = kernel[:,:,i]
        result = conv_per_channel(f_map, w, padding, stride)
        outputs += result

    return outputs

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default="img.jpg", help='path to input')
    parser.add_argument('--F', type=int, default=3, help='filter size')
    parser.add_argument('--C', type=int, default=3, help='filter channel')
    parser.add_argument('--K', type=int, default=2, help='filter num')
    parser.add_argument('--S', type=int, default=2, help='stride')
    parser.add_argument('--P', type=int, default=1, help='padding')
    return parser

def main(arg):
    path = Path(arg.img)
    img = Image.open(path)
    plt.axis("off")
    plt.imshow(img)
    plt.show()
    outputs = []
    input = np.array(img)
    print(f"Input: {input.shape}")
    kernel = np.random.randn(arg.K, arg.F, arg.F, arg.C)
    for i in range(arg.K):
        result = conv_per_kernel(input, kernel[i], arg.P, arg.S)
        outputs.append(result)


    return np.array(outputs).transpose(1,2,0)

if __name__ == '__main__':
    arg = get_arg_parser().parse_args()
    output = main(arg)
    print(f"Output: {output.shape}")



