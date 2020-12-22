import glob
import cv2

template = cv2.imread('Map3.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 99, 200)
cv2.imwrite('res.png',template)