import cv2
import imutils
import numpy as np
import keras
import tensorflow as tf
from sudoku_solver import sol
from skimage.segmentation import clear_border
import easyocr
class sudoku:

    def __init__(self,path):
        self.image=cv2.imread(path)
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
        cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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

    def Detect_Image(self,image):
        model = tf.keras.models.load_model('digit_model.h5')
        board = np.zeros((9, 9), dtype="int")
        # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
        # infer the location of each cell by dividing the warped image
        # into a 9x9 grid
        stepX = image.shape[1] // 9
        stepY = image.shape[0] // 9
        # initialize a list to store the (x, y)-coordinates of each cell
        # location
        cellLocs = []
        js = 0
        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []
            for x in range(0, 9):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                # add the (x, y)-coordinates to our cell locations list
                row.append((startX, startY, endX, endY))
                image=cv2.rectangle(image,(startX,startY),(endX,endY),(255,34,233),2)
                #cutting the image into Region Of Image
                cut=image[startX:endX,startY:endY]

                cut=cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
                cut=cv2.GaussianBlur(cut, (7, 7), 3)
                cut=cv2.adaptiveThreshold(cut, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                cut=cv2.bitwise_not(cut)
                cut=clear_border(cut)

                edged = cv2.Canny(cut, 30, 200)
                counter=cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(counter)

                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

                cv2.drawContours(cut, cnts, -1, (0, 255, 0), 3)
                cut = cv2.resize(cut, (28, 28), interpolation=cv2.INTER_AREA)
                #cv2.imwrite("/Users/macbook/PycharmProjects/IMPProject/images/{}.png".format(str(js)), cut)
                print("/Users/macbook/PycharmProjects/IMPProject/images/{}.png".format(str(js)))


                cellLocs.append(cut)
                cut=np.array(cut)
                cut=np.expand_dims(cut,axis=-1)
                cut=np.expand_dims(cut,axis=0)
                #print(cut.shape)
                print(cv2.contourArea(cnts[0]))
                if cv2.contourArea(cnts[0])>500:

                    ans=model.predict(cut).argmax()
                    ans=str(ans)
                    print(ans)
                    image = cv2.putText(image,ans, ((startX+endX)//2, (startY+endY)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),2)

                js += 1
        print(js)

        return image
    def number_reader(self,image):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image)
        for i in result:
            image=cv2.rectangle(image,(i[0][0][0],i[0][0][1]),(i[0][2][0],i[0][2][1]),(255,34,233),2)
        return image
    def detect(self,image):

        stepX = image.shape[1] // 9
        stepY = image.shape[0] // 9
        # initialize a list to store the (x, y)-coordinates of each cell
        # location
        cellLocs = []

        reader = easyocr.Reader(['en'])
        result=[]
        sudoku=[[],[],[],[],[],[],[],[],[]]

        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []
            js=0
            for x in range(0, 9):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                image = cv2.rectangle(image, (startX, startY), (endX, endY), (255, 34, 233), 2)
                cut=image[startX:endX,startY:endY]
                result=[]
                try:
                    result = reader.readtext(cut)
                    print(result)
                except:
                    print("image not loaded correctly")
                # if j==20:
                #     cv2.imshow("cut",cut)
                #     cv2.waitKey(0)

                if result!=[]:
                    image = cv2.putText(image, result[0][-2], ((startY + endY) // 2,(startX + endX) // 2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    sudoku[js].append(int(result[0][-2]))

                else:
                    image = cv2.putText(image,'0', ((startY + endY) // 2, (startX + endX) // 2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    sudoku[js].append(0)


                js+=1

        print(sudoku)

        return image,sudoku
    def solve(self,sudoku):
        s=sol()

        s.print_board(sudoku)
        s.solve(sudoku)
        print("___________________")
        s.print_board(sudoku)



s=sudoku("city_times.jpeg")
image=s.grayscale()
image=s.GaussianBlur()
image=s.threshold()
cnts=s.contours()

puzzlecnt=s.ROIpoints()

image=s.tranformation()

image=cv2.resize(image,(900,900),interpolation=cv2.INTER_AREA)
# image=s.Detect_Image(np.array(image))

# #image=cv2.resize(image,(400,400),interpolation=cv2.INTER_AREA)
#
#
# #image=test_fun(image)

#image=cv2.imread("city_times.jpeg")
#image = cv2.putText(image, i[-2], ((i[0][0][0]+i[0][2][0])//2, (i[0][0][1]+i[0][2][1])//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#cv2.drawContours(image,puzzlecnt,0,(255,34,23),5)
image,sudoku=s.detect(image)
s.solve(sudoku)
cv2.imshow("Tranformation of images 19BCE2086",image)
cv2.waitKey(0)