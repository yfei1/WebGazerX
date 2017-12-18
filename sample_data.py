import random
import csv
import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np


cutoff=50
#read in gaze coordinate and scale by *100
data=np.loadtxt(open('tobiigazerDataShuffle.csv', 'r'), delimiter=",")
data=data*100

#find max value of coordinate for x and y
x_max=np.max(data[range(0,data.shape[0]),0])#return 115
y_max=np.max(data[range(0,data.shape[0]),1])#return 173
x=data[range(0,data.shape[0]),0]
y=data[range(0,data.shape[0]),1]

#for heatmap
bins=np.zeros(shape=(12,18))

#save the bin label for each gaze
mydict=np.zeros(len(x), dtype=int)
for i in xrange(0,12):
	for j in xrange(0,18):
 		interval_x_l=10*i
 		interval_x_r=10*(i+1)
 		interval_y_l=10*j
 		interval_y_r=10*(j+1)
 		temp1=np.where(x>=interval_x_l)[0]
 		temp2=np.where(x<interval_x_r)[0]
 		temp3=np.where(y>=interval_y_l)[0]
 		temp4=np.where(y<interval_y_r)[0]
 		temp_index=np.intersect1d(np.intersect1d(temp1,temp2),np.intersect1d(temp3,temp4))
 		bins[i,j]=len(temp_index)
 		mydict[temp_index]=i*18+j
 		#for k in range(0, len(temp_index)):
 		#	mydict[temp_index[k]]=[i,j]

#save the number of gaze for each bin as 1D
hist=np.zeros(12*18)
for i in range(0,12*18):
	hist[i]=len(np.where(mydict==i)[0])

myindex=np.zeros(len(x))
bin_x_keep=np.where(bins<=cutoff)[0]
bin_y_keep=np.where(bins<=cutoff)[1]
for i in range(0, len(bin_x_keep)):
	dict_index=bin_x_keep[i]*18+bin_y_keep[i]
	myindex[np.where(mydict==dict_index)[0]]=1


bin_x=np.where(bins>cutoff)[0]
bin_y=np.where(bins>cutoff)[1]
for i in range(0, len(bin_x)):
	dict_index=bin_x[i]*18+bin_y[i]
	myindex[np.random.choice(np.where(mydict==dict_index)[0],cutoff, replace=False)]=1

##all the index
myrows=np.where(myindex==1)[0]


new_data=data[myrows]/100
np.savetxt("SampledtobiigazerShuffleData.csv", new_data, delimiter=",")
#print how many data you are using
print(len(myrows))


data1=np.loadtxt(open('eyeDataShuffle.csv', 'r'), delimiter=",")
newdata1=data1[myrows]
np.savetxt("SampledeyeData.csv", data1, delimiter=",")

data1=np.loadtxt(open('faceDataShuffle.csv', 'r'), delimiter=",")
newdata1=data1[myrows]
np.savetxt("SampledfaceShuffleData.csv", data1, delimiter=",")



data1=np.loadtxt(open('webgazerDataShuffle.csv', 'r'), delimiter=",")
newdata1=data1[myrows]
np.savetxt("SampledwebgazeShuffleData.csv", data1, delimiter=",")

