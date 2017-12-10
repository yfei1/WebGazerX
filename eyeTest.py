import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
from flask import Flask

class NN(nn.Module):

    def __init__(self, batch_size, hidden_size):
        super(NN, self).__init__()

        # conv2d input should (batch_size x channels x height x width)
        self.ey_norm1 = nn.InstanceNorm2d(128)
        self.ey_norm2 = nn.InstanceNorm2d(256)
        self.ey_norm3 = nn.InstanceNorm2d(96)
        self.fc_norm1 = nn.InstanceNorm1d(76)
        
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
        self.dropout5 = nn.Dropout(p=0.15)
        self.dropout6 = nn.Dropout(p=0.5)

        
        self.fc_w1 = nn.Linear(76, hidden_size) #batch_size x hidden_size
        self.final_w1 = nn.Linear(hidden_size*3, 2) #batch_size x2

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

        fc = self.fc_norm1(fc)
        fc = nn.ReLU()(self.fc_w1(fc))
        fc = self.dropout5(fc)

        gt_pred = torch.cat((ey,fc), dim = 1) # batch x (hidden_sizex3)
        gt_pred = self.dropout6(gt_pred)
        gt_pred = self.final_w1(gt_pred)
        return gt_pred.data
app = Flask(__name__)

@app.route('/')
def get_gazer_position():
    batch_size = 20
    hidden_size = 400

    ey = Variable(torch.rand(1, 2, 24, 48))
    fc = Variable(torch.rand(1, 76))

    startLoad = time.time()
    model = NN(batch_size, hidden_size)
    model.load_state_dict(torch.load("model"))
    endLoad = time.time()


    model.train(False)
    #model.eval()
    pred = model(ey, fc)
    endTest = time.time()
    print(pred[0])
    print("Load Time %f" % (endLoad-startLoad))
    print("Test Time %f" % (endTest-endLoad))
