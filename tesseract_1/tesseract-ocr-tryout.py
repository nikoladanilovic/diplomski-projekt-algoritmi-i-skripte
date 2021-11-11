import cv2 as cv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\ndanilovic\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

img = cv.imread('Slike\\Referent_picture_tv_guide.bmp')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#print(pytesseract.image_to_string(img))
#print(pytesseract.image_to_boxes(img))

### Detecting characters
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    print(b)
    b = b.split(' ')
    #print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv.rectangle(img, (x, hImg - y), (w, hImg - h), (0,0,255), 1)

cv.imshow('Result', img)
cv.imwrite('results of tesseract/boxes_of_detected_characters.bmp', img)
cv.waitKey(0)

