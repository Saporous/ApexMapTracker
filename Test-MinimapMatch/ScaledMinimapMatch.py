# Code by Adrian Rosebrock January 26, 2015
# https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--template", required=True, help="Path to template image")
# ap.add_argument("-i", "--images", required=True,
# 	help="Path to images where template will be matched")
# ap.add_argument("-v", "--visualize",
# 	help="Flag indicating whether or not to visualize each iteration")
# args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges
# template = cv2.imread(args["template"])
# template = cv2.imread('oly2.png')
template = cv2.imread('WE3.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 99, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)
# loop over the images to find the template in
# for imagePath in glob.glob(args["images"] + "/*.jpg"):
lastMaxVal = 0
iter = 0
mapName = ["World's Edge","Olympus"]
lastMapVal = 0
for imagePath in glob.glob('Map[0-9].png'):
# for imagePath in glob.glob('Olympus.png'):
	# load the image, convert it to grayscale, and initialize the
	# bookkeeping variable to keep track of the matched region
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
	scaleVal = None
	tempResult = None
	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 99, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		# check to see if the iteration should be visualized
		# if args.get("visualize", False):
		# 	# draw a bounding box around the detected region
		# 	clone = np.dstack([edged, edged, edged])
		# 	cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
		# 		(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
		# 	cv2.imshow("Visualize", clone)
		# 	cv2.waitKey(0)
		# if we have found a new maximum correlation value, then update
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
			scaleVal = scale
			tempResult = result
	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	# print(scaleVal)
	maxValRound = round(maxVal*100,2)
	# print(maxValRound)
	# print(startX)
	# print(startY)

	# draw a bounding box around the detected result and display the image
	# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	# fontFace=cv2.FONT_HERSHEY_SIMPLEX
	# cv2.putText(image, str(maxValRound),(startX, startY-4),fontFace,1,(0,0,255),2)
	# cv2.imshow("Image", image)
	# cv2.waitKey(0)
	if maxValRound > lastMaxVal:
		lastMaxVal = maxValRound
		lastMapVal = iter
	iter += 1

print("My best guess is the map is " + mapName[lastMapVal] 
+ " with " + str(lastMaxVal) + "% confidence")