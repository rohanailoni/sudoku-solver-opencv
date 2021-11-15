import cv2
import imutils
import numpy as np
from s1_fun import test_fun


class sudoku:

    def __init__(self,path):
        #self.image=cv2.imread(path)
        self.image=path
    #for converting the image into gray scale
    def grayscale(self):
        self.gray=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        return self.gray
    #appliying the gaussian blur on the image
    def GaussianBlur(self):
        self.blur=cv2.GaussianBlur(self.gray,(7,7),3)

        return self.blur


    #using adaptive Threshold because a paper may contain wide range of pixels
    def threshold(self):
        self.thresh=cv2.adaptiveThreshold(self.blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        self.thresh=cv2.bitwise_not(self.thresh)

        return self.thresh
    #finding contours
    def contours(self):
        cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        self.cnts=cnts

        return  cnts
    #getting the coordinates ROI
    def ROIpoints(self):
        self.puzzlecnt=[]
        for c in self.cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ROI = self.image[y:y + h, x:x + w]

                self.puzzlecnt.append(approx)
                break
        return self.puzzlecnt

    def order_points(self,pts):
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

    def tranformation(self):
        image=self.image
        try:
            points=np.array(self.puzzlecnt).reshape(4,2)
            rect = self.order_points(points)
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
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            # return the warped image
            return warped
        except ValueError:
            print("point the sudoku to the camera")
            return image
        except:
            print("Something else went wrong")

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)
        s = sudoku(frame)
        image = s.grayscale()
        image = s.GaussianBlur()
        image = s.threshold()
        s.contours()
        s.ROIpoints()
        image = s.tranformation()

        # image=test_fun(image)
        cv2.imshow("image", image)
        #cv2.waitKey(0)



        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
