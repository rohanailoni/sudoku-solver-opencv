import cv2
import imutils
import numpy as np
from s1_fun import test_fun,fun
image=cv2.imread("city_times.jpeg")

#converting the matrix into gray scale
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#adding gaussinan blur to blur the image
blurred = cv2.GaussianBlur(gray, (7, 7), 3)
#using adaptive threshold to convert image into binary
thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.bitwise_not(thresh)
#finding counters


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts=imutils.grab_contours(cnts)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
puzzleCnt=[]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ROI = image[y:y + h, x:x + w]


        puzzleCnt.append(approx)
        break


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect
def tranformation(image,points):
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth -1, 0],
        [maxWidth-1 , maxHeight-1],
        [0, maxHeight-1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

output = image.copy()
new_image=tranformation(image,np.array(puzzleCnt).reshape(4,2))
gray_transform=tranformation(gray,np.array(puzzleCnt).reshape(4,2))
test_fun(gray_transform)
cv2.drawContours(output, puzzleCnt, -1, (0, 255, 0), 2)
cv2.imshow("Puzzle Outline", gray_transform)





# image3=fun(ROI)
# cv2.imshow('new_image',image3)

# lol=sudoku_cells(new_image)
# cv2.imshow('modda',lol)
cv2.waitKey(0)

