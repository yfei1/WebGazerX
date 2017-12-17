import glob
import os
import csv
import cv2
import numpy as np
from pathlib import Path

def imgDistance(left, right):
    leftHist  = cv2.calcHist([left] , [0], None, [10], [0.0, 255.0])
    rightHist = cv2.calcHist([right], [0], None, [10], [0.0, 255.0])
    return np.sum(np.absolute(leftHist-rightHist))

def validEye(clmEyeMean, mean, std):
    return clmEyeMean < mean + std and clmEyeMean > mean - std

def getEyeRegion(clmTracker):
    clmLeftEyeLB = clmTracker[46]
    clmLeftEyeRB = clmTracker[50]
    clmLeftEyeTB = clmTracker[49]
    clmLeftEyeBB = clmTracker[53]
    
    clmRightEyeLB = clmTracker[60]
    clmRightEyeRB = clmTracker[56]
    clmRightEyeTB = clmTracker[59]
    clmRightEyeBB = clmTracker[63]

    clmLeftEye  = img[clmLeftEyeTB -3 : clmLeftEyeBB +3, clmLeftEyeLB -3 : clmLeftEyeRB +3]
    clmRightEye = img[clmRightEyeTB-3 : clmRightEyeBB+3, clmRightEyeLB-3 : clmRightEyeRB+3]

    if len(clmLeftEye) == 0 or len(clmRightEye) == 0:
        return [], []

    clmLeftEye  = cv2.equalizeHist(clmLeftEye)
    clmRightEye = cv2.equalizeHist(clmRightEye)
   
    clmLeftEye  = cv2.resize(clmLeftEye , (48, 24))
    clmRightEye = cv2.resize(clmRightEye, (48, 24))

    clmLeftEye  = clamp(clmLeftEye , 20, 245)
    clmRightEye = clamp(clmRightEye, 20, 245) 

    return clmLeftEye, clmRightEye

def clamp(img, lowerBound, upperBound):
    for row in img:
        for i in row:
            if i < lowerBound:
                i = 0
            elif i > upperBound:
                i = 255
    return img


eyeFile = open('eyeData.csv', 'w')
faceFile = open('faceData.csv', 'w')
webgazerFile = open('webgazerData.csv', 'w')
tobiigazerFile = open('tobiigazerData.csv', 'w')

eyeFileWriter = csv.writer(eyeFile, delimiter=',')
faceFileWriter = csv.writer(faceFile, delimiter=',')
webgazerFileWriter = csv.writer(webgazerFile, delimiter=',')
tobiigazerFileWriter = csv.writer(tobiigazerFile, delimiter=',')

levelOneViewList = sorted(glob.glob(r'./P*'))
#levelOneViewList = ['./P_1']
dataFile = "/gazePredictions.csv"

for levelOneDir in levelOneViewList:
    levelTwoViewList = sorted(glob.glob(levelOneDir + r'/*'))
    # levelTwoViewList = glob.glob(levelOneDir + r'/1493230037782_7*')
    # print(levelTwoViewList)
    count = 0
    invalid_count = 0

    perPersonMeanLeft = list()
    perPersonMeanRight = list()
    accImgDistMean = list()
        
    for dirToView in levelTwoViewList:
        filePath = Path(dirToView + dataFile)
        if filePath.is_file():
            print("Processing " + dirToView)
            with open( dirToView + dataFile ) as f:
                readCSV = csv.reader(f, delimiter=',')
                for row in readCSV:
                    frameFilename = row[0]
                    frameTimestamp = row[1]
                    
                    img = cv2.imread(frameFilename, cv2.IMREAD_GRAYSCALE)
                    imgHeight, imgWidth = img.shape
                    facePos = list()

                    # Tobii has been calibrated such that 0,0 is top left and 1,1 is bottom right
                    tobiiLeftEyeGazeX  = float(row[2])
                    tobiiLeftEyeGazeY  = float(row[3])
                    tobiiRightEyeGazeX = float(row[4])
                    tobiiRightEyeGazeY = float(row[5])

                    webgazerX = float(row[6])
                    webgazerY = float(row[7])

                    clmTracker = row[8:len(row)-1]
                    clmTracker = [float(i) for i in clmTracker]
                    clmTracker = [int(i)   for i in clmTracker]

                    clmLeftEye, clmRightEye = getEyeRegion(clmTracker)

                    if len(clmLeftEye) == 0 or len(clmRightEye) == 0:
                        invalid_count = invalid_count+1
                        continue

                    mLeft  = np.mean(clmLeftEye)
                    mRight = np.mean(clmRightEye)
                    dist   = imgDistance(clmLeftEye, clmRightEye)

                    if \
                            len(perPersonMeanLeft) > 0 and \
                            validEye(mLeft , np.mean(perPersonMeanLeft) , 1.5*np.std(perPersonMeanLeft) ) and \
                            validEye(mRight, np.mean(perPersonMeanRight), 1.5*np.std(perPersonMeanRight)) and \
                            np.absolute(dist - np.mean(accImgDistMean)) < 1.5*np.std(accImgDistMean): #TODO: Detect if blinking or not
                        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
                        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

                        # Jaw
                        facePos.extend(clmTracker[0:30])

                        # Upper lip
                        facePos.extend(clmTracker[88:102])

                        # Lower lip
                        facePos.extend(clmTracker[102:112])

                        # Eye position
                        facePos.extend(clmTracker[126:142])

                        # Nose
                        facePos.extend(clmTracker[66:68])
                        facePos.extend(clmTracker[82:84])
                        facePos.extend(clmTracker[124:126])
                        #cv2.imshow('face', img)
                        #cv2.waitKey(0)
                        #cv2.imshow('leftEye', clmLeftEye)
                        #cv2.waitKey(0)
                        #cv2.imshow('rightEye', clmRightEye)
                        #cv2.waitKey(0)
                        #print("Duila!")
                        # len of facePos is 76
                        for i in range(len(facePos)):
                            if i % 2 == 0:
                                facePos[i] = facePos[i]*(1600/imgWidth)
                            else:
                                facePos[i] = facePos[i]*(900/imgHeight)


                        eyeFileWriter.writerow(np.concatenate((clmLeftEye, clmRightEye)).flatten())
                        faceFileWriter.writerow(facePos)
                        webgazerFileWriter.writerow([webgazerX, webgazerY])
                        tobiigazerFileWriter.writerow([tobiiEyeGazeX, tobiiEyeGazeY])
                    else:
                        invalid_count = invalid_count+1
                        #cv2.imshow('leftEye', clmLeftEye)
                        #cv2.imshow('rightEye', clmRightEye)
                        #cv2.waitKey(0)
                        #print(count)
                    perPersonMeanLeft.append(mLeft)
                    perPersonMeanRight.append(mRight)
                    accImgDistMean.append(dist)
                    count = count + 1
        else:
            print(str(filePath) + " does not exists")
    invalid_rate = str(round(invalid_count/count*100, 2))+"%" if count is not 0 else str(0)+"%"
    print(levelOneDir + " with " + str(count) + " files, invalid file rate " + invalid_rate)

eyeFile.close()
faceFile.close()
webgazerFile.close()
tobiigazerFile.close()
