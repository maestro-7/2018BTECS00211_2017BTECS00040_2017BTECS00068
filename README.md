# Number Plate Detection
 * 2017BTECS00040
 * 2017BTECS00068
 * 2018BTECS00211
 
## Algorithm :
1) Take a car image and convert it to grey scale image.
2) Apply noise reduction algorithm and histogram equalization.
3) Generate morphological opening output and subtract from Histogram Equalization output.
4) Apply thresholding and pass output to Canny edge detector.
5) Find contours using edges.
6) Show anding output of contour with original image.
7) Enhance Output.

Hence Number Plate is Located

## Video Demo 
 
https://drive.google.com/open?id=1RWA9tpCBI3OPlCC_enzSV9vz4oml6bxF
