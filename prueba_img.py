import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import datetime

cara_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#'E:\openCV\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,400)
frame = cv2.imread('img/ejemplo1.png')

gris = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
edged_frame = cv2.Canny(gris,100,120)
cv2.imshow('Canny detección de borde',edged_frame)
cv2.imwrite('img/prueba_edged_frame1.jpg',edged_frame)

denoised = cv2.GaussianBlur(gris,(5,5),0)
filtro = cv2.Laplacian(gris,cv2.CV_8U)#64F)
filtro2 = cv2.Laplacian(denoised,cv2.CV_8U)#64F)
cv2.imshow('Laplacian Filter',filtro)
cv2.imwrite('img/prueba_Laplacian1.jpg',filtro)
cv2.imshow('Laplacian Filter denoised (+Gaussian Blur)',filtro2)
cv2.imwrite('img/prueba_Laplacian_denoised1.jpg',filtro2)
cv2.imshow('Original',frame)
cv2.waitKey(0)


cv2.destroyAllWindows()



# 
# img = frame
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 5, 50, 50)
# edges = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
# edges_high_thresh = cv2.medianBlur(gray,1)
# #edges = cv2.Sobel(gray,cv2.CV_8U,1,1)
# edges = cv2.Canny(edges,120,160,apertureSize = 3,L2gradient=True)
# #edges = cv2.boxFilter(gray, 0, (7,7), img, (-1,-1), False, cv2.BORDER_DEFAULT)
# #edges = cv2.medianBlur(edges,1)
# edges_high_thresh = cv2.Canny(edges_high_thresh,50,100,apertureSize = 3,L2gradient=True)
# 
# images = np.hstack((edges, edges_high_thresh))
# cv2.imshow('Canny edges',images)
# cv2.imwrite('img/prueba_Canny_edges.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# linesHT = cv2.HoughLinesP(edges_high_thresh,1,np.pi/180,100,minLineLength,maxLineGap)
# for (x1,y1,x2,y2) in lines[0,:]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# for (x1,y1,x2,y2) in linesHT[0,:]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# images = np.hstack((gray, edges, edges_high_thresh))
# #cv2.namedWindow('Canny edges', cv2.WINDOW_NORMAL)
# cv2.imshow('Canny edges',images)
# cv2.imshow('original',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('img/prueba_houghlines5.jpg',img)