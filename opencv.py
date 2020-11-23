# import cv2 as cv

# img = cv.imread("img1.jpg")
# cv.namedWindow("image") # 创建一个image的窗口
# cv.imshow("image", img) # 显示图像
# cv.waitKey(0)            # 默认为0， 无限等待
# cv.destroyAllWindows()  # 释放窗口

# img = cv.imread("img1.jpg")
# cv.imwrite('./img/img1.jpg', img)

# img = cv.imread("img1.jpg")
# b, g, r = cv.split(img)  # 拆分图像通道为b、g、r三个通道
# cv.imshow("b", b)
# cv.imshow("g", g)
# cv.imshow("r", r)
# cv.imshow("image", img)
# cv.waitKey(0)
# cv.destoryAllWindows()

# img = cv.imread("img1.jpg")
# b, g, r = cv.split(img)
# imgagebgr = cv.merge([b, g, r])
# cv.imshow("img", img)
# cv.imshow("imgagebgr", imgagebgr)
# cv.waitKey()
# cv.destroyAllWindows()


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

import cv2 as cv

image1 = cv.imread("img1.jpg")
cv.imshow("image", image1)
image2 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
image2 = cv.cvtColor(image2, cv.COLOR_RGB2HSV)
cv.imshow("image2", image2)
cv.waitKey()
cv.destroyAllWindows()


