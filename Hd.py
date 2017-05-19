import sys
import os
import numpy as np
import h5py 
import opt
import math
import NavHd
import TrajHd

#yaw,x,y,speed,(x,y,yaw,velocity)
#define actions
sys.path.append('/home/sofia96/Downloads/trajactory-learning/script')
from Myclass import const
const.LEFT=0
const.RIGHT=2
const.STRAIGHT=1
const.ACCELERATE=0
const.STABLE=1
const.REDUCTION=2

def show_state_in_files(filename,l):
	f=open(filename,"w")
	for line in l:
		f.write(str(line)[1:-1]+"\n")
	f.close()

#set array's precision to acc
def set_array_precision(array,acc):
	b=np.ones((array.shape[0],array.shape[1]))
	for i in range(0,array.shape[0]):
		for j in range(0,array.shape[1]):
			b[i][j]=round(array[i][j],acc)
			print(b[i][j])
	return b

#initialize h5File and create two groups
Traj=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/2013122701.h5","w")
trajectory=Traj.create_group(".TRAJ")
nav=Traj.create_group(".NAV")
state=Traj.create_group(".STAT")
action=Traj.create_group(".ACT")  
Traj.close()

#write traj and nav to5 h5
TrajHd.Traj_handling("/home/sofia96/Downloads/trajactory-learning/traj1-l/2013122701001.traj")

#join together ego and ob cars
Traj=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/2013122701.h5","r+")
nav=Traj[".NAV"]
trajactory=Traj[".TRAJ"]
name=Traj.create_group(".STAT/001")
Traj.create_group(".ACT/001")
egoarray=np.array(nav["001/ego"])
obs=trajactory["001"]
i=0
obsdict={}
for name in obs:
	obsdict[i]=np.array(obs[name])
	i=i+1
	print(i)

arrayrow=egoarray.shape[0]
stateobservationarray=np.zeros((arrayrow,24))
for index in range(0,arrayrow):
	stateobservationarray[index][0:4]=egoarray[index][1:5]
	stateobservationarray[index][4:8]=obsdict[0][index][1:5]
	stateobservationarray[index][8:12]=obsdict[1][index][1:5]
	stateobservationarray[index][12:16]=obsdict[2][index][1:5]
	stateobservationarray[index][16:20]=obsdict[3][index][1:5]
	stateobservationarray[index][20:24]=obsdict[4][index][1:5]

#set float to .00
stateobservationarray=set_array_precision(stateobservationarray,2)

#get the action sequence
actionobservationarray=np.zeros((arrayrow-1,2))
for i in range(0,arrayrow-1):
	if stateobservationarray[i+1][0]-stateobservationarray[i][0]>0.005:
		actionobservationarray[i][0]=const.LEFT
	elif stateobservationarray[i+1][0]-stateobservationarray[i][0]<-0.005:
		actionobservationarray[i][0]=const.RIGHT
	else:
		actionobservationarray[i][0]=const.STRAIGHT
	if stateobservationarray[i+1][3]-stateobservationarray[i][3]>0.01:
		actionobservationarray[i][1]=const.ACCELERATE
	elif stateobservationarray[i+1][3]-stateobservationarray[i][3]<-0.01:
		actionobservationarray[i][1]=const.REDUCTION
	else:
		actionobservationarray[i][1]=const.STABLE

show_state_in_files("20131201001.sta",list(stateobservationarray))
show_state_in_files("20131201001.act",list(actionobservationarray))

Traj.create_dataset(".STAT/001/state",data=stateobservationarray)
Traj.create_dataset(".ACT/001/action",data=actionobservationarray)
Traj.close()
