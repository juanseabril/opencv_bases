import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2 import rectangle
from cv2 import bitwise_and


img = cv.imread('Dataset/3.jpg')

blank = np.zeros(img.shape[:2], dtype='uint8') #Img vacia del mismo tamaño que la original

#img_resize = cv.resize(img, (500, 500))

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

"""
img_canny = cv.Canny(img, 125, 175) #Detecta bordes

img_dilated = cv.dilate(img_canny, (3,3), iterations=10)

img_eroded = cv.erode(img_dilated, (3,3), iterations=10) #Aumenta pix. negros

ret, thresh = cv.threshold(img_gray, 125, 255, cv.THRESH_BINARY) 

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(blank, contours, -1, (0, 0, 255), 2 ) 
cv.imshow('Fondo De Ojo', blank)
cv.waitKey(0)
"""

"""
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # Matiz, saturación, brillo

img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB) # Matiz, saturación, brillo

b, g, r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
"""

"""
rectangle = cv.rectangle(blank.copy(), (30, 30), (650, 570), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

bitwise_And = cv.bitwise_and(rectangle, circle)

bitwise_Or = cv.bitwise_or(rectangle, circle)

bitwise_Xor = cv.bitwise_xor(rectangle, circle) 

bitwise_Not = cv.bitwise_not(rectangle)

masked = cv.bitwise_and(img, img, mask=rectangle)
"""

"""
circle = cv.circle(blank.copy(), (200,150), 100, 255, -1)
masked = cv.bitwise_and(img, img, mask=circle)
cv.imshow("original", masked)

plt.figure()
plt.title("Histograma")
plt.xlabel("valor")
plt.ylabel("# pixeles")
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
     hist = cv.calcHist([img], [i], circle, [256], [0,256])
     plt.plot(hist, color=col)
     plt.xlim([0, 256])

plt.show()
cv.waitKey(0)
"""

"""
threshold, thresh = cv.threshold(img_gray, 125, 255, cv.THRESH_BINARY) 
cv.imshow('smple', thresh)

adaptive_thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('adapt', adaptive_thresh)

cv.waitKey(0)
"""

lap = cv.Laplacian(img_gray, cv.CV_64F) #Gradiente
lap = np.uint8(np.absolute(lap)) #Absoluto porque no pueden haber pix negativos
#cv.imshow('Lapaclian', lap)

sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0)#Gradiente en dos direcciones
sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('sobelx', sobelx)
cv.imshow('sobely', sobely)
cv.imshow('combine', combined_sobel)
cv.waitKey(0)
"""
NOTAS: 

~ Las imagenes son BGR
~ Si realizo una difuminación antes de calcular los contornos voy a obtener menos.
~ FindContours puede ser por canny o por thresh
~ FindContours-thresh sirve para detectar en que lugar se encuentra la luz focal
~ Matplotlib tiene como formato imagenes RGB
~ Bitwise operaciones AND + XOR = OR
~ En el thresh adaptive el valor de C puede ser 0 y es el resultado de la media
~ En el thresh adaptive se puede usar el valor gaussiano en vez del promedio
"""