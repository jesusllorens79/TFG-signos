import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

print(cv2.__version__)
print(np.__version__)

cam = cv2.VideoCapture(0) #ID de la cámara
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "./Data/Z"
counter = 0

while True:
    success, img = cam.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize,imgSize, 3), np.uint8)*255 # de 0 a 255, tipico de imagenes
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w +offset]
        
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize / h
            wCalculated = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCalculated, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCalculated)/2)
            imgWhite[:, wGap:wCalculated+wGap] = imgResize
            
        else:
            k = imgSize / w
            hCalculated = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCalculated))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCalculated)/2)
            imgWhite[hGap:hCalculated+hGap,:] = imgResize
            
        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1) 
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)