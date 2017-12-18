import glob
import csv
import cv2
import numpy as np
from pathlib import Path
import dlib
from imutils import face_utils

def imgDistance(left, right):
    leftHist  = cv2.calcHist([left] , [0], None, [10], [0.0, 255.0])
    rightHist = cv2.calcHist([right], [0], None, [10], [0.0, 255.0])
    return np.sum(np.absolute(leftHist-rightHist))

def validEye(clmEyeMean, mean, std):
    return clmEyeMean < mean + std and clmEyeMean > mean - std

# This is a clmTracker based face & eye detector
def getEyeRegionCLM(clmTracker):
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

def get_jaw(shape, face_features):
    for (x, y) in shape[0:15]:
        face_features.append(x)
        face_features.append(y)


def get_lip(shape, face_features):
    for (x, y) in shape[48:60]:
        face_features.append(x)
        face_features.append(y)


def get_eye(shape, face_features):
    eye_pos = [37, 38, 40, 41, 44, 43, 47, 45]
    for i in eye_pos:
        for (x, y) in shape[i:i+1]:
            face_features.append(x)
            face_features.append(y)


def get_nose(shape, face_features):
    for (x, y) in shape[28:31]:
        face_features.append(x)
        face_features.append(y)


def getEyeRegionDlib(shape, frame):
    leftEyeLB = shape[36][0]
    leftEyeRB = shape[39][0]
    leftEyeTB = int((shape[37][1] + shape[38][1] - 5)/2)
    leftEyeBB = int((shape[40][1] + shape[41][1] + 5)/2)

    rightEyeLB = shape[42][0]
    rightEyeRB = shape[45][0]
    rightEyeTB = int((shape[43][1] + shape[44][1] - 5)/2)
    rightEyeBB = int((shape[46][1] + shape[47][1] + 5)/2)

    leftEye  = frame[leftEyeTB -3 : leftEyeBB +3, leftEyeLB -3 : leftEyeRB +3]
    rightEye = frame[rightEyeTB-3 : rightEyeBB+3, rightEyeLB-3 : rightEyeRB+3]

    if len(leftEye) == 0 or len(rightEye) == 0:
        return [], []

    leftEye  = cv2.equalizeHist(leftEye)
    rightEye = cv2.equalizeHist(rightEye)

    leftEye  = cv2.resize(leftEye , (48, 24))
    rightEye = cv2.resize(rightEye, (48, 24))

    leftEye  = clamp(leftEye , 20, 245)
    rightEye = clamp(rightEye, 20, 245)

    return np.concatenate((leftEye, rightEye)).flatten()


def clamp(img, lowerBound, upperBound):
    for row in img:
        for i in row:
            if i < lowerBound:
                i = 0
            elif i > upperBound:
                i = 255
    return img


useDlibTracker = True

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

                    if useDlibTracker:
                        detector = dlib.get_frontal_face_detector()
                        predictor = dlib.shape_predictor('.\dlib-models\shape_predictor_68_face_landmarks.dat')
                        gray = img
                        faces = detector(gray, 1)
                        if len(faces) == 0:
                            invalid_count = invalid_count+1
                            continue

                        for (_, rect) in enumerate(faces):
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

                            face_features = list()
                            eye_features = list()

                            eye_region = getEyeRegionDlib(shape, gray)

                            if len(eye_region) != 2304:
                                invalid_count = invalid_count+1
                                continue

                            get_jaw(shape, face_features)
                            get_lip(shape, face_features)
                            get_eye(shape, face_features)
                            get_nose(shape, face_features)

                            for i in range(0, len(face_features), 2):
                                cv2.circle(img, (int(face_features[i]), int(face_features[i+1])), 2, (0, 0, 255), -1)

                            for i in range(len(face_features)):
                                if i % 2 == 0:
                                    face_features[i] = face_features[i]*(1600/imgWidth)
                                else:
                                    face_features[i] = face_features[i]*(900/imgHeight)

                            eye_features = [float(i) for i in eye_region]
                            face_features = [float(i) for i in face_features]

                            eyeFileWriter.writerow(eye_features)
                            faceFileWriter.writerow(facePos)
                            webgazerFileWriter.writerow([webgazerX, webgazerY])
                            tobiigazerFileWriter.writerow([tobiiEyeGazeX, tobiiEyeGazeY])
                    else:
                        clmTracker = row[8:len(row)-1]
                        clmTracker = [float(i) for i in clmTracker]
                        clmTracker = [int(i)   for i in clmTracker]

                        clmLeftEye, clmRightEye = getEyeRegionCLM(clmTracker)

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
