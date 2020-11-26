# import cv2 as cv

# img = cv.imread("img1.jpg")
# cv.namedWindow("image") # 创建一个image的窗口
# cv.imshow("image", img) # 显示图像
# cv.waitKey(0)            # 默认为0， 无限等待
# cv.destroyAllWindows()  # 释放窗口


# import cv2 as cv
# img = cv.imread("img1.jpg")
# cv.imwrite('./img/img1.jpg', img)


# import cv2 as cv
# img = cv.imread("img1.jpg")
# b, g, r = cv.split(img)  # 拆分图像通道为b、g、r三个通道
# cv.imshow("b", b)
# cv.imshow("g", g)
# cv.imshow("r", r)
# cv.imshow("image", img)
# cv.waitKey(0)
# cv.destoryAllWindows()

# import cv2 as cv
# img = cv.imread("img1.jpg")
# b, g, r = cv.split(img)
# imgagebgr = cv.merge([b, g, r])
# cv.imshow("img", img)
# cv.imshow("imgagebgr", imgagebgr)
# cv.waitKey()
# cv.destroyAllWindows()



# import cv2 as cv
# img = cv.imread("img1.jpg")
# print("img.shape", img.shape)  # 输出图像的大小属性
# print("img.size", img.size)  # 输出图像的像素数目属性
# print("img.dtype", img.dtype) # 输出图像的类型属性


# import cv2 as cv
# import numpy as np

# imagegray = np.random.randint(0, 256, size=[256, 256], dtype=np.uint8)
# cv.imshow("imagegray", imagegray)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import numpy as np

# image1 = np.random.randint(0, 256, size=[4, 4], dtype=np.uint8)
# image2 = np.random.randint(0, 256, size=[4, 4], dtype=np.uint8)

# print("image1=\n", image1)
# print("image2=\n", image2)
# print("image3=\n", image1 + image2)

# import numpy as np

# image1 = np.random.randint(0, 256, size=[4, 4], dtype=np.uint8)
# image2 = np.random.randint(0, 256, size=[4, 4], dtype=np.uint8)
# print("image1=\n", image1)
# print("image2=\n", image2)
# print("image3=\n", image1 - image2)


# import numpy as np

# arr1 = np.random.randint(0, 256, size=[3, 4], dtype=np.uint8)
# arr2 = np.random.randint(0, 256, size=[4, 3], dtype=np.uint8)
# arr3 = np.dot(arr1, arr2)
# print("arr1=\n", arr1)
# print("arr2=\n", arr2)
# print("arr3=\n", arr3)


# import numpy as np
# import cv2 as cv

# image1 = np.random.randint(0, 256, size=[4, 4], dtype=np.uint8)
# image2 = np.random.randint(0, 256, size=[4, 4], dtype=np.uint8)
# image3 = cv.divide(image1, image2)
# print("image1=\n", image1)
# print("image2=\n", image2)
# print("image3=\n", image3)


# import cv2 as cv
# import numpy as np

# image1 = cv.imread("img1.jpg")
# cv.imshow("image", image1)
# image2 = np.zeros(image1.shape, dtype=np.uint8)  # 构造掩模图像
# image2[100:400, 100:400] = 255
# image3 = cv.bitwise_and(image1, image2)  # 进行按位与。取出掩模内的图像
# cv.imshow("image3", image3)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv

# image1 = cv.imread("img1.jpg")
# cv.imshow("image", image1)
# image3 = cv.bitwise_not(image1)  # 按位非操作图像取反
# cv.imshow("image3", image3)
# cv.waitKey()
# cv.destroyAllWindows()

# import cv2 as cv

# image1 = cv.imread("img1.jpg")
# cv.imshow("image", image1)
# image2 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)  # BGR色彩空间转RGB色彩空间
# cv.imshow("image2", image2)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv

# image1 = cv.imread("img1.jpg")
# cv.imshow("image", image1)

# image2 = cv.cvtColor(image1, cv.COLOR_BGR2RGB) # BGR色彩空间转化RGB色彩空间
# image2 = cv.cvtColor(image2, cv.COLOR_RGB2GRAY)  # RGB色彩空间转GRAY色彩空间
# cv.imshow("image2", image2)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv

# image1 = cv.imread("img1.jpg")
# cv.imshow("image", image1)
# image2 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
# image2 = cv.cvtColor(image2, cv.COLOR_RGB2YCrCb)

# cv.imshow("image2", image2)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv

# image1 = cv.imread("img1.jpg")
# cv.imshow("image", image1)
# image2 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
# image2 = cv.cvtColor(image2, cv.COLOR_RGB2HSV)
# cv.imshow("image2", image2)
# cv.waitKey()
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np

# image = cv.imread("img1.jpg")
# h, w = image.shape[:2] # 获取图像大小信息
# M = np.float32([[1, 0, 120], [0, 1, -120]]) # 构建转换矩阵
# imageMove = cv.warpAffine(image, M, (w, h))  # 进行仿射变换---平移
# cv.imshow("image", image)
# cv.imshow("imageMove", imageMove)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv
# import numpy as np

# image = cv.imread("img1.jpg")
# h, w = image.shape[:2]
# M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
# imageMove = cv.warpAffine(image, M, (w, h))
# cv.imshow("image", image)
# cv.imshow("imageMove", imageMove)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# image = cv.imread("img1.jpg")
# h, w = image.shape[:2]
# # 得到转换矩阵M,效果是以图像的宽高的1/3为中心顺时针旋转40度。缩小为原来的0.4
# M = cv.getRotationMatrix2D((w / 3, h / 3), 40, 0.4)
# imageMove = cv.warpAffine(image, M, (w, h))
# cv.imshow("image", image)
# cv.imshow("imageMove", imageMove)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv
# import numpy as np

# image = np.random.randint(0, 256, size=[6, 6], dtype=np.uint8)
# w, h = image.shape  # 得到数组的宽与高
# # 建立新数组的大小
# x = np.zeros((w, h), np.float32)
# y = np.zeros((w, h), np.float32)

# # 实现新数组的访问操作
# for i in range(w):
#     for j in range(h):
#         x.itemset((i, j), j)
#         y.itemset((i, j), i)

# rst = cv.remap(image, x, y, cv.INTER_LINEAR)  # 实现数组的复制
# print("image=\n", image)
# print("rst=\n", rst)


# import cv2 as cv
# import numpy as np

# image = cv.imread("img1.jpg")
# w, h = image.shape[:2]
# map1 = np.zeros((w, h), np.float32)
# map2 = np.zeros((w, h), np.float32)
# # 实现新图像的访问操作
# for i in range(w):
#     for j in range(h):
#         map1.itemset((i, j), j)
#         map2.itemset((i, j), w - 1 - i)
# rst = cv.remap(image, map1, map2, cv.INTER_LINEAR)
# cv.imshow("image", image)
# cv.imshow("rst", rst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np

# image = cv.imread("img1.jpg")
# w, h = image.shape[:2]  # 得到图像的宽与高
# map1 = np.zeros((w, h), np.float32)
# map2 = np.zeros((w, h), np.float32)

# # 实现新图像的访问操作
# for i in range(w):
#     for j in range(h):
#         map1.itemset((i, j), h - 1 - j)
#         map2.itemset((i, j), i)

# rst = cv.remap(image, map1, map2, cv.INTER_LINEAR)
# cv.imshow("image", image)
# cv.imshow("rst", rst)
# cv.waitKey(0)
# cv.destroyAllWindows()



# import cv2 as cv
# import numpy as np

# image = cv.imread("img1.jpg")
# h, w = image.shape[:2]
# src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float32)
# dst = np.array([[80, 80], [w / 2, 50], [80, h - 80], [w - 40, h - 40]], np.float32)
# # 计算投影变换矩阵
# M = cv.getPerspectiveTransform(src, dst)
# # 进行投影变换
# image1 = cv.warpPerspective(image, M, (w, h), borderValue=125)
# cv.imshow("image", image)
# cv.imshow("image1", image1)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv

# image = cv.imread("img1.jpg", cv.IMREAD_ANYCOLOR)
# dst = cv.linearPolar(image, (251, 249), 225, cv.INTER_LINEAR)

# cv.imshow("image", image)
# cv.imshow("dst", dst)

# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv

# image = cv.imread("img1.jpg")
# # 计算其统计直方图信息
# hist = cv.calcHist([image], [0], None, [256], [0, 255])
# print(hist)

# import cv2 as cv
# import matplotlib.pyplot as plt

# arr1 = [1, 1.2, 1.5, 1.6, 2, 2.5, 2.8, 3.5, 4.3]
# arr2 = [5, 4.5, 4.3, 4.2, 3.6, 3.4, 3.1, 2.5, 2.1, 1.5]

# plt.plot(arr1)
# plt.plot(arr2, 'r')
# plt.show()


# import cv2 as cv
# import matplotlib.pyplot as plt 
# image = cv.imread('img1.jpg')
# hist = cv.calcHist([image], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()



# import cv2 as cv
# import matplotlib.pyplot as plt 

# image = cv.imread("img1.jpg")
# image = image.ravel() # 将图像转化为一维数组
# plt.hist(image, 256) # 绘制直方图
# cv.waitKey(0)
# cv.destroyAllWindows()
# plt.show()


# import numpy as np
# import cv2 as cv
# import math
# import matplotlib.pyplot as plt 

# # 计算图像灰度直方图
# def calcGrayHist(image):
#     # 灰度图像的矩阵的宽高
#     rows = image.shape[0]
#     cols = image.shape[1] 
#     # 存储灰度直方图
#     grayHist = np.zeros([256], np.uint32)
#     for r in range(rows):
#         for c in range(cols):
#             grayHist[image[r][c]] += 1
#     return grayHist

# # 直方图均衡化
# def equalHist(image):
#     rows = image.shape[0]
#     cols = image.shape[1]
    
#     grayHist = calcGrayHist(image)
#     # 计算累加直方图
#     zeroCumuMoment = np.zeros([256], np.uint32)
#     for p in range(256):
#         if p == 0:
#             zeroCumuMoment[p] = grayHist[0]
#         else:
#             zeroCumuMoment[0] = zeroCumuMoment[p - 1] + grayHist[p]
#     # 根据直方图均衡化得到饿输入灰度和输出灰度的映射
#     outPut_q = np.zeros([256], np.uint8)
#     cofficent = 256.0 / (rows * cols)
#     for p in range(256):
#         q = cofficent * float(zeroCumuMoment[p]) - 1
#         if q >= 0:
#             outPut_q[p] = math.floor(q)
#         else:
#             outPut_q[p] = 0
#     # 得到直方图均衡化的图像
#     equalHistImage = np.zeros(image.shape, np.uint8)
#     for r in range(rows):
#         for c in range(cols):
#             equalHistImage[r][c] = outPut_q[image[r][c]]
#     return equalHistImage

# image = cv.imread("./img1.jpg")
# dst = equalHist(image) # 直方图均衡化
# cv.imshow("image", image)
# cv.imshow("dst", dst)

# # 显示原始图像直方图
# plt.figure("原始直方图")
# plt.hist(image.ravel(), 256)
# # 显示均衡化后的图像直方图
# plt.figure("均衡化直方图")
# plt.hist(dst.ravel(), 256)
# plt.show()
# cv.waitKey(0)
# cv.destroyAllWindows()

# import cv2 as cv

# image = cv.imread('./img1.jpg')
# cv.imshow("image", image)
# # 定义卷积5*5，采用自动计算权重的方式实现高斯滤波
# gauss = cv.GassianBlur(image, (5, 5), 0, 0)
# cv.imshow("gauss", gauss)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv

# image = cv.imread("./img1.jpg")
# # 定义一个卷积和为5*5， 实现均值滤波
# means5 = cv.blur(image, (5, 5))
# means10 = cv.blur(image, (10, 10))
# means20 = cv.blur(image, (20, 20))
# cv.imshow("means5", means5)
# cv.imshow("means10", means10)
# cv.imshow("means20", means20)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv
# image = cv.imread("./img1.jpg")
# ret, dst = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
# cv.imshow("image", image)
# cv.imshow("dst", dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

import cv2 as cv
image = cv.imread("./img1.jpg")
ret, dst = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)
cv.imshow("dst", dst)
cv.waitKey(0)
v.destroyAllWindows()




















