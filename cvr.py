import cv2 as cv
img = cv.imread('image (15).JPG')
img = cv.resize(img, (256,256))
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()