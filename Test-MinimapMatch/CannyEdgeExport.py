import glob
import cv2

fileName = "Map3"
template = cv2.imread(fileName+".png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 99, 200)
cv2.imwrite(fileName+"ED.png",template)