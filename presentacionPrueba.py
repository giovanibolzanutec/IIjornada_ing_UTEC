import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import datetime

cara_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#'E:\openCV\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,400)
ret = True
while(ret):
    ret, frame = cap.read()
    gris = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    edged_frame = cv2.Canny(gris,90,120)
    cv2.imshow('Canny detección de borde',edged_frame)
    denoised = cv2.GaussianBlur(gris,(5,5),0)
    filtro = cv2.Laplacian(gris,cv2.CV_8U)#64F)
    filtro2 = cv2.Laplacian(denoised,cv2.CV_8U)#64F)
    cv2.imshow('Laplacian Filter',filtro)
    cv2.imshow('Laplacian Filter denoised (+Gaussian Blur)',filtro2)
    listaCaras = cara_cascade.detectMultiScale(gris)
    for (x,y,w,h) in listaCaras:
        print("cara")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
        cv2.circle(frame,(x,y),2,(0,0,255),-1)
        cv2.putText(frame,"una cara",(x,y),1,2,(120,100,190),4) #font, 3,(211,211,211),4)
        #cv2.rectangle(x,y,w,h)
    cv2.imshow('Original',frame)
    frameFlipped = cv2.flip(frame,0)    # write the flipped frame
    cv2.imshow('Invertido',frameFlipped)
    k = cv2.waitKey(5) & 0xFF
    if k==27:   break
cap.release()
cv2.destroyAllWindows()