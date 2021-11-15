import cv2
import imutils
import numpy as np
#import keras
from skimage.segmentation import clear_border
#import asyncio

def fun(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # adding gaussinan blur to blur the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    cv2.imshow('blurred', blurred)
    # using adaptive threshold to convert image into binary
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    thresh = cv2.bitwise_not(thresh)
    # finding counters

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    puzzleCnt = []
    ROI = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ROI.append(image[y:y + h, x:x + w])
            if cv2.contourArea(approx) > 2000 and cv2.contourArea(approx) < 5000:
                puzzleCnt.append(approx)
    print(len(puzzleCnt))
    output = image.copy()
    cv2.drawContours(output, puzzleCnt, -1, (0, 255, 0), 2)
    return output


def test_fun(image):
    #model = keras.models.load_model('cnn.h5')
    board = np.zeros((9, 9), dtype="int")
    # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid

    stepX = image.shape[1] // 9
    stepY = image.shape[0] // 9
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []
    js=0
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

            cells = image[startY:endY, startX:endX]



            cellLocs.append(cells)


            image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 255), 5)

            #cell = cv2.adaptiveThreshold(cells, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            cell = cv2.threshold(cells, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            cell = clear_border(cell)

            cv2.imshow("name likewise", cell)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            #cell = cv2.bitwise_not(cell)

            # cell=cv2.GaussianBlur(cell,(7,7),3)

            mask = np.zeros(cell.shape, dtype="uint8")

            cnts = cv2.findContours(cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnts = imutils.grab_contours(cnts)
            if len(cnts) == 0:
                return None
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


            c = min(cnts, key=cv2.contourArea)

            mask = cv2.drawContours(mask, [c], -1, 255, -1)

            arc=cv2.arcLength(c,True)

            approx=cv2.approxPolyDP(c,0.02*arc,True)

            x,y,w,h=cv2.boundingRect(approx)

            test_sampl=cell[y:y+h,x:x+w]
            #cv2.imshow('test sample',test_sampl)

            mask = cv2.bitwise_not(mask)
            #cv2.imshow("this is binary cell", mask)

            img = cv2.resize(test_sampl, dsize=(28, 28))
            img=img.astype('float')/255.0

            img = np.array(img).reshape(-1, 28, 28, 1)
            break

            # j= extract_digits(img,model)
            #
            # image=cv2.putText(image,str(j),(startX,startY+endY),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
            # js+=1






def extract_digits(cells, model):
    just =model.predict(cells)

    just = np.argmax(just)

    return just
