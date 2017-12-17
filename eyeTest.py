import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import cv2
import dlib
import json
from imutils import face_utils
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, current_app
from flask_socketio import SocketIO, emit
import base64


class NN(nn.Module):

    def __init__(self, batch_size, hidden_size):
        super(NN, self).__init__()

        # conv2d input should (batch_size x channels x height x width)
        self.ey_norm1 = nn.InstanceNorm2d(128)
        self.ey_norm2 = nn.InstanceNorm2d(256)
        self.ey_norm3 = nn.InstanceNorm2d(96)

        self.ey_conv1 = nn.Conv2d(2, 128, (3,3), padding=1) #24*48
        self.ey_conv2 = nn.Conv2d(128, 256, (3,3), padding=1) #8*16
        self.ey_conv3 = nn.Conv2d(256, 96, (3,3), padding=1) 

        self.ey_maxpool1 = nn.MaxPool2d(3) #batch_size x 4 x 8 x 16
        self.ey_maxpool2 = nn.MaxPool2d(4) #batch_size x 4 x 2 x 4
        self.ey_w1 = nn.Linear(96*2*4, hidden_size*2) #batch_size x (hidden_sizex2)
        
        self.dropout1 = nn.Dropout(p=0.15)
        self.dropout2 = nn.Dropout(p=0.15)
        self.dropout3 = nn.Dropout(p=0.15)        
        self.dropout4 = nn.Dropout(p=0.15)
        #self.dropout5 = nn.Dropout(p=0.15)
        #self.dropout6 = nn.Dropout(p=0.5)

        
        #self.fc_w1 = nn.Linear(76, hidden_size) #batch_size x hidden_size
        self.final_w1 = nn.Linear(hidden_size*2, 2) #batch_size x2

    def forward(self, ey, fc):
        ey = nn.ReLU()(self.ey_conv1(ey))
        ey = self.ey_maxpool1(ey)
        ey = self.ey_norm1(ey)
        ey = self.dropout1(ey)

        ey = nn.ReLU()(self.ey_conv2(ey))
        ey = self.ey_maxpool2(ey)
        ey = self.ey_norm2(ey)
        ey = self.dropout2(ey)

        ey = nn.ReLU()(self.ey_conv3(ey))
        ey = self.ey_norm3(ey)
        ey = self.dropout3(ey)

        ey = ey.view(1,-1)
        ey = nn.ReLU()(self.ey_w1(ey))
        ey = self.dropout4(ey)

        #fc = self.fc_norm1(fc)
        #fc = nn.ReLU()(self.fc_w1(fc))
        #fc = self.dropout5(fc)

        #gt_pred = torch.cat((ey,fc), dim = 1) # batch x (hidden_sizex3)
        #gt_pred = self.dropout6(gt_pred)
        gt_pred = self.final_w1(ey)
        return gt_pred.data

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


def get_eye_region(shape, frame):
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


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@socketio.on('webcam-image')
def get_gazer_position(web_json):
    img = web_json['image']
    clientTime = int(web_json['clientTime'])
    clickX = float(web_json['clickX'])
    clickY = float(web_json['clickY'])

    if time.time() - clientTime > 20:
        return "Server is busy"

    # decode client webcam image into numpy uint8 array
    img = base64.b64decode(img.split(',')[1])
    img = np.array(bytearray(img), dtype=np.uint8)
    frame = cv2.imdecode(img, -1)

    x = list()
    y = list()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgHeight, imgWidth = gray.shape
    faces = current_app.detector(gray, 1)

    if len(faces) == 0:
        return "别歪头快开灯"

    for (_, rect) in enumerate(faces):
        shape = current_app.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        face_features = list()
        eye_features = list()

        eye_region = get_eye_region(shape, gray)

        if len(eye_region) != 2304:
            return "别歪眼，快开灯"

        get_jaw(shape, face_features)
        get_lip(shape, face_features)
        get_eye(shape, face_features)
        get_nose(shape, face_features)

        for i in range(0, len(face_features), 2):
            cv2.circle(frame, (int(face_features[i]), int(face_features[i+1])), 2, (0, 0, 255), -1)

        for i in range(len(face_features)):
            if i % 2 == 0:
                face_features[i] = face_features[i]*(1600/imgWidth)
            else:
                face_features[i] = face_features[i]*(900/imgHeight)

        eye_features = [float(i) for i in eye_region]
        face_features = [float(i) for i in face_features]

        ey = Variable(torch.FloatTensor(eye_features)).view(1, 2, 24, 48)
        fc = Variable(torch.FloatTensor(face_features)).view(1, 76)

        startTest = time.time()
        pred = model(ey, fc)
        endTest = time.time()
        modelGazePoint = pred[0].numpy()
        x.append(modelGazePoint[0])
        y.append(modelGazePoint[1])
        #modelGazePoint[0] = modelGazePoint[0]*(imgHeight/600)
        #modelGazePoint[1] = modelGazePoint[1]*(imgWidth/900)

        print("Test Time %f" % (endTest-startTest))

    #plt.scatter(x, y)
    #plt.show()
    res_json = dict()
    res_json['numOfPeople'] = len(faces)
    res_json['eyePredictions'] = list()

    for i in range(len(x)):
        curPred = dict()
        curPred['x'] = int(x[i])
        curPred['y'] = int(y[i])
        res_json['eyePredictions'].append(curPred)

    print(res_json)
    emit("prediction", json.dumps(res_json))


if __name__ == '__main__':
    batch_size = 20
    hidden_size = 400

    ctx = app.app_context()
    ctx.push()

    startLoad = time.time()
    model = NN(batch_size, hidden_size)
    model.load_state_dict(torch.load("onlyeye"))
    endLoad = time.time()
    model.train(False)
    model.eval()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('.\dlib-models\shape_predictor_68_face_landmarks.dat')


    current_app.model = model
    current_app.predictor = predictor
    current_app.detector = detector

    print("Load Time %f" % (endLoad-startLoad))
    socketio.run(app, host='0.0.0.0', debug=True, port=5000)
#get_gazer_position()
