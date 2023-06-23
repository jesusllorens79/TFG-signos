import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier

print(cv2.__version__)
print(np.__version__)

cap = cv2.VideoCapture(0) # ID de la cámara
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\jesus\Desktop\tfg_signos\Model\tres_keras_model.h5", 
                        r"C:\Users\jesus\Desktop\tfg_signos\Model\tres_keras_model.txt")
offset = 20
imgSize = 300
counter = 0
phrases = []
current_phrase = ""
labels = ["A","B","C", "D", "E" ,"F" ,"G" ,"H" ,"I", "J" ,"K" ,"L", "M" ,"N" ,"O" ,"P" ,"Q" ,"R" ,"S" ,"T" ,"U" ,"V", "W" ,"X" ,"Y" ,"Z"]


while True:
    success, img = cap.read()
    if img is None:
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        if cv2.waitKey(1) & 0xFF == ord('v'):  # Espera a que se presione la tecla "v"
            current_phrase += labels[index]

        if cv2.waitKey(1) & 0xFF == ord(' '):  # Espera a que se presione la tecla "espacio"
            current_phrase += " "  # Agrega un espacio a la frase actual

        if cv2.waitKey(1) & 0xFF == ord('s'):  # Espera a que se presione la tecla "s" para guardar la frase actual
            phrases.append(current_phrase)
            current_phrase = ""

    # Dibujar frases en la imagen
    y_offset = 100
    for phrase in phrases:
        cv2.putText(imgOutput, phrase, (50, y_offset), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
        y_offset += 50

    cv2.putText(imgOutput, current_phrase, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
