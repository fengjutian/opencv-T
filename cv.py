# import cv2 as cv
# img = cv.imread("IMG_2022.JPG")
# print(img)


#import cv2 as cv

# img = cv.imread("IMG_2022.JPG")
# cv.namedWindow("image")
# cv.imshow("image", img)
# cv.waitKey()
# cv.destroyAllWindows()


# import cv2 as cv
# import numpy as np 

# imagegray = np.random.randint(0, 256, size=[256, 256], dtype=np.uint8)
# cv.imshow("imagegray", imagegray)
# cv.waitKey(0)
# cv.destroyAllWindows()


import cv2 as cv
import numpy as np

img = np.random.randint(0, 256, size=[256, 256, 3], dtype=np.uint8)
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()


