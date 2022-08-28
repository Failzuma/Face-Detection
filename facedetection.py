import cv2 as cv

cascadeModel = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
cascadeWajah = cv.CascadeClassifier(cascadeModel)
gambar = cv.imread('./img/blackpink.jpg')
ubahkeGray = cv.cvtColor(gambar, cv.COLOR_BGR2GRAY)
wajah = cascadeWajah.detectMultiScale(ubahkeGray, 1.29,6)
for (x,y,w,h) in wajah:
    cv.rectangle(gambar,(x,y), (x+w, y+h), (255,255,0),2)
cv.imshow('Deteksi Wajah',gambar)
cv.waitKey()