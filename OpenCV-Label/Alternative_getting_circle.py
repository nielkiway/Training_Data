import cv2
import numpy as np

img = cv2.imread('GridTest.tif')
gray = cv2.imread('GridTest.tif', cv2.IMREAD_GRAYSCALE)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,130,255,0)
#ret, frame_threshold = cv2.threshold(img_2, 130, 255, 0)
#thresh = cv2.bitwise_not(thresh)
cv2.imshow("img", thresh)
cv2.waitKey()

element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))

morph_img = thresh.copy()
cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)

contours,_ = cv2.findContours(morph_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
sorted_areas = np.sort(areas)

#bounding box (red)
cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour
#r = cv2.boundingRect(cnt)
#cv2.rectangle(img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),2)

#min circle (green)
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(img,center,radius,(0,255,0),1)

print(int(x), int(y), radius)

##fit ellipse (blue)
#ellipse = cv2.fitEllipse(cnt)
#cv2.ellipse(img,ellipse,(255,0,0),1)


cv2.imshow("morph_img",morph_img)
cv2.imshow("img", img)
cv2.waitKey()

#enter image description here