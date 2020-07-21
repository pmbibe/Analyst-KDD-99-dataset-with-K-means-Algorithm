import numpy as np
import pandas
from time import time
import matplotlib.pyplot as plt
import collections, numpy
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, zero_one_loss, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def process_file(file,startTime, stopTime):
    recvdSize = 0
    t_put = 0
    s = 0
    r =0
    with open(file) as data_file:
        for line in data_file:
            if line.split(" ")[7] == "cbr":
                event = line.split(" ")[0]
                time = line.split(" ")[1]
                time = float(time)
                pkt_size = line.split(" ")[8]
                pkt_size = float(pkt_size)
                level = line.split(" ")[3]
                if level == "AGT" and (startTime < time < stopTime) :
                    if event == "s":
                        s = s+1
                    elif event == "r":
                        r = r + 1

        try:
            return (float(r/s)*100)

        except:
            pass


t= []
time = 0
while time < 100:
    t.append(process_file("out_Normal.tr", time, time+10))
    time = time+10
time=[0, 10,20,30,40,50,60,70,80,90]

t1= []
time1 =0
while time1 < 100:
    t1.append(process_file("out_Attack_10s.tr", time1, time1+10))
    time1 = time1+10
time1=[0, 10,20,30,40,50,60,70,80,90]

t2= []
time2 = 0
while time2 < 100:
    t2.append(process_file("out_Attack.tr", time2, time2+10))
    time2 = time2+10
time2=[0, 10,20,30,40,50,60,70,80,90]
plt.plot(time2, t2, label = "Blackhole attack")
plt.plot(time1, t1, label = "Blackhole attack after 10s")
plt.plot(time, t, label = "Normal")
plt.xlabel('Time')
plt.ylabel('Packet Delivery Ratio')
plt.legend()
plt.show()

