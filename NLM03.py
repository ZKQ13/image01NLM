import cv2
import numpy as np

def double2unit8(I, L, ratio=1.0, sigma=18.0):
    I = I.astype(np.float64)  # 转换成float形式便于计算
    noise = np.random.randn(*I.shape) * sigma  # 生成的时形状与I相同，均值为0，标准差为sigam的随机高斯噪声矩阵
    noisy = I + noise  # 噪声图像
    return np.clip(np.round(noisy * ratio), 0, 255).astype(L.dtype)  # 转换回图像的unit8均值

I = cv2.imread('yuantu.jpg', 0)  # 以灰度图像形式读取
sigma = 20.0
I1 = double2unit8(I, I, sigma=20.0)  # 对图像加上高斯噪声
R2 = cv2.fastNlMeansDenoising(I1, None, sigma, 5, 11)  # 利用opencv自带的NLM去噪
cv2.imshow("Image", I)#原图
cv2.imshow("Noisy", I1)#噪声图像
cv2.imshow("CVNLM", R2)#去噪图像
cv2.waitKey(0)  # 显示图像必备
cv2.destroyALLWindows()  # 释放窗口