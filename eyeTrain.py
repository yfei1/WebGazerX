from collections import OrderedDict
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import csv
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from argparse import ArgumentParser
import codecs, random, math
import torch.nn.functional as F

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

    def forward(self, ey, fc, gt, batch_size):
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

        ey = ey.view(batch_size, -1)
        ey = nn.ReLU()(self.ey_w1(ey))
        ey = self.dropout4(ey)        
        
        #fc = self.fc_norm1(fc)
        fc = nn.ReLU()(self.fc_w1(fc))
        fc = self.dropout5(fc)

        gt_pred = torch.cat((ey,fc), dim = 1) # batch x (hidden_sizex3)
        gt_pred = self.dropout6(gt_pred)
        gt_pred = self.final_w1(gt_pred)
        loss = (gt_pred - gt).pow(2).sum()
        return loss


def test_acc(ey, fc, gt, wg, model):
    batch_size = 100
    model.train(True)
    total_loss = 0
    total_benchmark_loss = 0

    for it in range(0, len(ey) - len(ey) % batch_size, batch_size):
        batch_ey = ey[it:it+batch_size]
        batch_fc = fc[it:it+batch_size]
        batch_gt = gt[it:it+batch_size]
        batch_wg = wg[it:it+batch_size]

        total_benchmark_loss = total_benchmark_loss + error(batch_gt, batch_wg, batch_size)
        batch_ey = Variable(torch.FloatTensor(batch_ey)).cuda()
        batch_fc = Variable(torch.FloatTensor(batch_fc)).cuda()
        batch_gt = Variable(torch.FloatTensor(batch_gt)).cuda()

        loss = model(batch_ey, batch_fc, batch_gt, batch_size)
        total_loss = total_loss + loss.data[0]

    model.train(True)
    return total_loss/(len(ey) - len(ey)%batch_size), total_benchmark_loss/(len(ey) - len(ey)%batch_size)

def error(gt, wg, batch_size):
    assert(len(gt) == len(wg))
    assert(len(gt[0]) == len(wg[0]))

    gt = np.array(gt)
    wg = np.array(wg)

    return pow((gt-wg), 2).sum()



batch_size = 20
hidden_size = 400

tobii_f = open('tobiigazerDataShuffle.csv', 'r')
eye_f   = open('eyeDataShuffle.csv'       , 'r')
face_f  = open('faceDataShuffle.csv'      , 'r')
web_f   = open('webgazerDataShuffle.csv'  , 'r')

print("Reading Files...")

gt = [[float(i)*100 for i in row] for row in csv.reader(tobii_f)]
ey = [[float(i)    for i in row] for row in csv.reader(eye_f)  ]
fc = [[float(i)    for i in row] for row in csv.reader(face_f) ]
wg = [[float(i)*100 for i in row] for row in csv.reader(web_f)  ]

assert(len(gt) == len(ey) == len(fc) == len(wg))
allData = list(zip(ey, fc, gt, wg))

model = NN(batch_size, hidden_size)
model = model.cuda()
best_err_so_far = 2000

parameters = list(model.parameters())
with open('parameters.txt', 'w') as f:
    f.write(str(parameters))


optimizer = optim.Adam(model.parameters(), lr = 1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.98)

testing_files = 5000
testData  = allData[:testing_files]
testData  = list(zip(*testData))

testEyData = np.array(list(testData[0])).reshape(testing_files, 2, 24, 48)
testFcData = list(testData[1])
testGtData = list(testData[2])
testWgData = list(testData[3])

trainData = allData[testing_files:]

print("Training Starts...")
print("Training Data %d" % len(trainData))
print("Testing Data %d" % len(testFcData))

start_time = time.time()

for epoch_num in range(120):
    training_err = 0
    for it in range(0, len(trainData)-len(trainData)%batch_size, batch_size):
        batchData = list(zip(*trainData[it:it+batch_size]))
        ey_batch = np.array(list(batchData[0])).reshape(batch_size, 2, 24, 48)
        fc_batch = list(batchData[1])
        gt_batch = list(batchData[2]) # ground truth batch
        wg_batch = list(batchData[3])

        gt_batch = Variable(torch.FloatTensor(gt_batch)).cuda()
        fc_batch = Variable(torch.FloatTensor(fc_batch)).cuda()
        ey_batch = Variable(torch.FloatTensor(ey_batch)).cuda()

        loss = model(ey_batch, fc_batch, gt_batch, batch_size)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        training_err = training_err + loss.data[0]

        if it % 2000 == 0:
            print("%d : Loss = %.4f" % (it, loss.data[0]/batch_size))

    scheduler.step()
    lr = scheduler.get_lr()[0]
    err, benchmark_err = test_acc(testEyData, testFcData, testGtData, testWgData, model)
    print("%d th training" % epoch_num)
    print("Learning rate: %.6f" % lr)
    print("Validation Err: %2.3f" % err)
    print("Benchmark Err: %2.3f" % benchmark_err)
    print("Training Err: %2.3f" % (training_err/(len(trainData)-len(trainData)%batch_size)) )
    if best_err_so_far > err:
        best_err_so_far = err
        torch.save(model.state_dict(), "model")

print("Training took %f seconds" % (time.time()-start_time))
print("Best Validation Error so far: %2.3f" % best_err_so_far)
#torch.save(model.state_dict(), "model")
tobii_f.close()
eye_f.close()
face_f.close()
web_f.close()
