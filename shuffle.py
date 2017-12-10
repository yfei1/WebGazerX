import random
import csv

ey_f = open('eyeData.csv', 'r')
fc_f = open('faceData.csv', 'r')
gt_f = open('tobiigazerData.csv', 'r')
wg_f = open('webgazerData.csv', 'r')

new_ey = open('eyeDataShuffle.csv', 'w')
new_fc = open('faceDataShuffle.csv', 'w')
new_gt = open('tobiigazerDataShuffle.csv', 'w')
new_wg = open('webgazerDataShuffle.csv', 'w')

ey = [row for row in ey_f]
fc = [row for row in fc_f]
gt = [row for row in gt_f]
wg = [row for row in wg_f]

assert(len(ey) == len(fc) == len(gt) == len(wg))

allData = list(zip(ey, fc, gt, wg))
random.shuffle(allData)

for eye, face, tobii, web in allData:
    new_ey.write(eye)
    new_fc.write(face)
    new_gt.write(tobii)
    new_wg.write(web)

ey_f.close()
fc_f.close()
gt_f.close()
wg_f.close()

new_ey.close()
new_fc.close()
new_gt.close()
new_wg.close()
