import cv2                 
import numpy as np          


img = cv2.imread('./pictures/Car.jpg')                   
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)     
cv2.imshow("Original Image",img)                       

# RGB to Gray scale conversion
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.namedWindow("Filter 1 - Grayscale Conversion",cv2.WINDOW_NORMAL)
cv2.imshow("Filter 1 - Grayscale Conversion",img_gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
cv2.namedWindow("Filter 2 - Noise Removal(Bilateral Filtering)",cv2.WINDOW_NORMAL)
cv2.imshow("Filter 2 - Noise Removal(Bilateral Filtering)",noise_removal)

# Histogram equalisation for better results
equal_histogram = cv2.equalizeHist(noise_removal)
cv2.namedWindow("Filter 3 - Histogram equalisation",cv2.WINDOW_NORMAL)
cv2.imshow("Filter 3 - Histogram equalisation",equal_histogram)

# Morphological opening with a rectangular structure element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))                               
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)     
cv2.namedWindow("Filter 4 - Morphological opening",cv2.WINDOW_NORMAL)
cv2.imshow("Filter 4 - Morphological opening",morph_image)

# Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
cv2.namedWindow("Filter 5 - Image Subtraction", cv2.WINDOW_NORMAL)
cv2.imshow("Filter 5 - Image Subtraction", sub_morp_image)

# Thresholding the image
ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
cv2.namedWindow("Filter 6 - Thresholding",cv2.WINDOW_NORMAL)
cv2.imshow("Filter 6 - Thresholding",thresh_image)

# Applying Canny Edge detection
canny_image = cv2.Canny(thresh_image,250,255)
cv2.namedWindow("Filter 7 - Canny Edge Detection",cv2.WINDOW_NORMAL)
cv2.imshow("Filter 7 - Canny Edge Detection",canny_image)

canny_image = cv2.convertScaleAbs(canny_image)

# Dilation - to strengthen the edges
kernel = np.ones((3,3), np.uint8)                              
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)    
cv2.namedWindow("Filter 8 - Dilation(closing)", cv2.WINDOW_NORMAL)
cv2.imshow("Filter 8 - Dilation(closing)", dilated_image)

contours, hierarchy= cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]

NumberPlateCnt = None

for c in contours:
     peri = cv2.arcLength(c, True)
     approx = cv2.approxPolyDP(c, 0.06 * peri, True) 
    
     
     if len(approx) == 4:           
          NumberPlateCnt = approx   
          break                    

# Drawing 
final = cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)

cv2.namedWindow("Filter 9 - Approximated Contour",cv2.WINDOW_NORMAL)
cv2.imshow("Filter 9 - Approximated Contour",final)


# SEPARATING OUT THE NUMBER PLATE 

mask = np.zeros(img_gray.shape,np.uint8)                            
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1,)       
new_image = cv2.bitwise_and(img,img,mask=mask)                     
cv2.namedWindow("Result 10 - Number Plate Separation",cv2.WINDOW_NORMAL)
cv2.imshow(" Result 10 - Number Plate Separation",new_image)



#HISTOGRAM EQUALIZATION 


y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))       
y = cv2.equalizeHist(y)                                                 
final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)    
cv2.namedWindow("Result 11 - Enhanced Number Plate",cv2.WINDOW_NORMAL)
cv2.imshow("Result 11 - Enhanced Number Plate",final_image)


cv2.waitKey()                                                           
