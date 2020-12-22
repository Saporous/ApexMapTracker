# https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/

import numpy as np 
import cv2 
   
      
# Read the query image as query_img 
# and traing image This query image 
# is what you need to find in train image 
# Save it in the same directory 
# with the name image.jpg   
query_img = cv2.imread('WE1.png') 
train_img = cv2.imread('Map2.png') 
   
# Convert it to grayscale 
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
# query_img_bw = cv2.Canny(query_img_bw, 99, 200)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
# train_img_bw = cv2.Canny(train_img_bw, 99, 200)
# Initialize the ORB detector algorithm 
orb = cv2.ORB_create(nfeatures=100000, scaleFactor=1.2) 
   
# Now detect the keypoints and compute 
# the descriptors for the query image 
# and train image 
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 
  
# Initialize the Matcher for matching 
# the keypoints and then match the 
# keypoints 
matcher = cv2.BFMatcher() 
matches = matcher.match(queryDescriptors,trainDescriptors) 
   
# draw the matches to the final image 
# containing both the images the drawMatches() 
# function takes both images and keypoints 
# and outputs the matched query image with 
# its train image 
final_img = cv2.drawMatches(query_img_bw, queryKeypoints,  
train_img_bw, trainKeypoints, matches[:20],None) 
   
final_img = cv2.resize(final_img, (2000,1300)) 
  
# Show the final image 
cv2.imshow("Matches", final_img) 
cv2.waitKey(0) 
