import cv2 as cv
import numpy as np

img = cv.imread('Slike/Referent_picture_tv_guide.bmp')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints_tv_guide_ref_1.jpg',img)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints_with_orientation_tv_guide_ref_1.jpg',img)

kp,des = sift.compute(gray,kp)

print(len(kp))  #number of key points

a_file = open("descriptors.txt", "w")
for row in des:
    np.savetxt(a_file, row)

a_file.close()

cv.imshow('Drvo', img)

cv.waitKey(0)