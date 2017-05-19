import numpy as np
import h5py
import sys
import math
import random

sys.path.append('/home/sofia96/Downloads/trajactory-learning/script')
from Myclass import const
const.SAMPLENUM=200
const.LEFT=0
const.RIGHT=2
const.STRAIGHT=1
const.ACCELERATE=0
const.STABLE=1
const.REDUCTION=2
#yaw,x,y,speed,(x,y,yaw,velocity)

def show_state_in_files(filename,l):
	f=open(filename,"w")
	for line in l:
		f.write(str(line)[1:-1]+"\n")
	f.close()

#randomly generate states
def random_state():
	l=[]
	l.append(random.uniform(0,6.28))
	l.append(random.uniform(-10,10))
	l.append(random.uniform(-10,10))
	l.append(random.uniform(0,30))
	for i in range(0,5):
		l.append(random.uniform(-10,10))
		l.append(random.uniform(-10,10))
		l.append(random.uniform(0,6.28))
		l.append(random.uniform(0,30))
	return l

#randomly generate actions
def random_action():
	direction=[const.LEFT,const.STRAIGHT,const.RIGHT]
	speed=[const.ACCELERATE,const.STABLE,const.REDUCTION]
	a=[]
	a.append(random.randint(0,2))
	a.append(random.randint(0,2))
	return a

def next_state_cal(s0,a0):
	s=[]
	ego_radius=s0[0]
	ego_x=s0[1]
	ego_y=s0[2]
	ego_velocity=s0[3]
	ego_vx=ego_velocity*math.cos(ego_radius)
	ego_vy=ego_velocity*math.sin(ego_radius)
	
	#randomly pick out radius accelerate
	if a0[0]==0:
		dif=random.uniform(0.001,0.01)
	elif a0[0]==2:
		dif=random.uniform(-0.01,-0.001)
	else:
		dif=random.uniform(-0.001,0.001)
	#randomly pick out accelerate
	if a0[1]==0:
		acc=random.uniform(0.01,0.2)
	elif a0[1]==2:
		acc=random.uniform(-0.2,-0.01)
	else:
		acc=random.uniform(-0.01,0.01)

	s.append(ego_radius+dif)
	s.append(ego_x+ego_velocity*math.cos(ego_radius)*0.05)
	s.append(ego_y+ego_velocity*math.sin(ego_radius)*0.05)
	s.append(ego_velocity+acc)
	for i in range(0,5):
		s.append(s0[4+4*i]+(s0[7+4*i]*math.sin(s0[6+4*i])-ego_vx)*0.05)
		s.append(s0[5+4*i]+(s0[7+4*i]*math.cos(s0[6+4*i])-ego_vy)*0.05)
		s.append(s0[6+4*i]+random.uniform(-0.01,0.01))
		s.append(s0[7+4*i]+random.uniform(-0.2,0.2))

	return s

#set h5file to save samples
sample=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/samples.h5","w")
sample.create_group("state0")
sample.create_group("action")
sample.create_group("state1")

state_0=[]
action=[]
state_1=[]
#randomly pick out initial action
for index in range(0,const.SAMPLENUM):
	s0=random_state()
	a0=random_action()
	s1=next_state_cal(s0,a0)
	state_0.append(s0)
	state_1.append(s1)
	action.append(a0)

#save to h5files
sample.create_dataset("state0/pysical",data=np.array(state_0))
sample.create_dataset("state1/pysical",data=np.array(state_1))
sample.create_dataset("action/pysical",data=np.array(action))

#load h5files
expert_traj=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/2013122701.h5","r")
states=np.array(expert_traj[".STAT/001/state"])
actions=np.array(expert_traj[".ACT/001/action"])
statelist=list(states)
actionlist=list(actions)
sampledstate0list=[]
sampledstate1list=[]
sampledactionlist=[]
statelen=len(statelist)

#randomly pick out states and actions
for index in range(0,int(statelen/3)):
	i=random.randint(0,statelen-2)
	sampledstate0list.append(statelist[i])
	sampledactionlist.append(actionlist[i])
	sampledstate1list.append(statelist[i+1])

show_state_in_files("state0.stat",sampledstate0list)
show_state_in_files("state1.stat",sampledstate1list)
show_state_in_files("action.act",sampledactionlist)

#load h5files
sample.create_dataset("state0/expert",data=np.array(sampledstate0list))
sample.create_dataset("state1/expert",data=np.array(sampledstate1list))
sample.create_dataset("action.expert",data=np.array(sampledactionlist))

sample.close()

