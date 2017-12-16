import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np

tobii_f = open('tobiigazerDataShuffle.csv', 'r')

gt = [[float(i)*100 for i in row] for row in csv.reader(tobii_f)]


hi = plt.figure()

x = list()
y = list()

heat = [[0 for i in range(100) ] for i in range(100)]

for row in range(len(gt)):
    x.append(gt[row][0])
    y.append(gt[row][1])

heatmap, xedges, yedges = np.histogram2d(x, y, bins=(100,100))
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap) #, extent = extent)
plt.show()

plt.savefig('./heatmap.png')    
