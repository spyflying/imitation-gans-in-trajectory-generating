import sys
import os
import numpy as np
import h5py 
import opt
import math
import opt
import NavHd
sys.path.append('/home/sofia96/Downloads/trajactory-learning/script')

#define several const integers
from Myclass import const
const.MAXEPX=10
const.MAXEPY=20
const.CARTRUNC=5

def show_traj_in_files(filename,dic):
	f=open(filename,"w")
	for key,value in dic.items():	
		f.write(str(key)+"\n")
		for line in value:
			f.write(str(line)[1:-1]+'\n')
	f.close()

def getminkey(d):
	v=list(d.values())
	k=list(d.keys())
	return k[v.index(min(v))]

def get_state_inf(l,index):
	epx=l[index][1]
	epy=l[index][2]
	radius=l[index][3]
	velocity=l[index][4]
	return epx,epy,radius,velocity


def position_cal(x,y,radius,velocity,rev):
	if rev:
		epx=x-math.sin(radius)*velocity*0.05
		epy=y-math.cos(radius)*velocity*0.05
	else:
		epx=x-math.cos(radius)*velocity*0.05
		epy=y-math.sin(radius)*velocity*0.05
	return epx,epy

def find_first_ele(l,value):
	for index in range(0,len(l)):
		if int(l[index][0])==value:
			return index
	return -1

def replace_float_to_int(l):
	for ele in l:
		i=int(ele[0])
		ele.remove(ele[0])
		ele.insert(0,i)
	return l

#normalize the dict elements to have the same length
def Traj_normalize(dic):
	#extract all the timestamps
	Timestamp=set()
	TimestampDic={}
	for key,val in dic.items():
		for ele in val:
			Timestamp.add(ele[0])

	#normalize timestamps with 50:24 or 74
	tmpTimestamp=list(Timestamp)
	for ele in Timestamp:
		if int(ele)%50!=24:
			tmpTimestamp.remove(ele)
	Timestamp=set(tmpTimestamp)
	minTimestamp=min(Timestamp)
	maxTimestamp=max(Timestamp)
	for index in range(int(minTimestamp),int(maxTimestamp),50):
		if index not in Timestamp:
			Timestamp.add(index)
	l=list(Timestamp)
	
	#use the ego info to emplement
	NavHd.Nav_normalize("/home/sofia96/Downloads/trajactory-learning/traj1-l/2013122701001.nav",sorted(l),"001")
	nav=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/2013122701.h5","r")
	ego=np.array(nav[".NAV/001/ego"])
	#resort dict's value over timestamp
	for key,val in dic.items():
		dic[key]=sorted(dic[key],key=opt.first_ele)

	#unify the timestamps
	for key,value in dic.items():
			for ele in value:
				if int(ele[0])%50!=24:
					dic[key].remove(ele)

	for key,val in dic.items():
		velocity=val[0][4]
		radius=val[0][3]
		firstTimestamp=val[0][0]
		ego_radius=ego[0][1]
		ego_velocity=ego[0][4]		
		#implement the front
		for index in range(int(firstTimestamp-50),int(minTimestamp-50),-50):
			epx=val[0][1]
			epy=val[0][2]
			vx=velocity*math.sin(radius)-math.cos(ego_radius)*ego_velocity
			vy=velocity*math.cos(radius)-math.sin(ego_radius)*ego_velocity
			dic[key].insert(0,[index,epx-vx*0.05,epy-vy*0.05,radius,velocity])

		velocity=val[-1][4]
		radius=val[-1][3]
		lastTimestamp=val[-1][0]
		ego_radius=ego[-1][1]
		ego_velocity=ego[-1][4]
		#inplement the tail
		for index in range(int(lastTimestamp),int(maxTimestamp),50):
			epx=val[-1][1]
			epy=val[-1][2]
			vx=velocity*math.sin(radius)-math.cos(ego_radius)*ego_velocity
			vy=velocity*math.cos(radius)-math.sin(ego_radius)*ego_velocity
			dic[key].append([index,epx+vx*0.05,epy+vy*0.05,radius,velocity])
		
		
	show_traj_in_files("testtimestamp.test",dic)

	for key,val in dic.items():
		#implement the others
		ThisTimestamp=[]
		for ele in val:
			ThisTimestamp.append(int(ele[0]))
		for ele in Timestamp:
			if int(ele) not in ThisTimestamp:
				print(ele)
				l=dic[key]
				print(ele-50)
				PreviousIndex=find_first_ele(dic[key],ele-50)
				print(PreviousIndex)
				epx,epy,radius,velocity=get_state_inf(dic[key],PreviousIndex)
				dic[key].insert(PreviousIndex+1,[ele,epx+(math.sin(radius))*0.05,epy+(math.cos(radius))*0.05,radius,velocity])
				print(PreviousIndex+1,[ele,epx+(math.sin(radius))*0.05,epy+(math.cos(radius))*0.05,radius,velocity])
		print("*********************************")
	
	return dic

#save normalized trajectories to h5
def Save_traj_to_h5(dic):
	f=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/2013122701.h5","r+")
	traj=f['.TRAJ']
	thisTraj=traj.create_group("001")
	for key,val in dic.items():
		t0=thisTraj.create_dataset(str(key),data=np.array(val))
		t0.attrs["tno"]=key
	f.close()


#initialize h5File and create two groups
Traj=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/2013122701.h5","w")
trajectory=Traj.create_group(".TRAJ")
nav=Traj.create_group(".NAV")

#open the file
Traj=open("/home/sofia96/Downloads/trajactory-learning/traj1-l/2013122701001.traj","r")
NormedTraj=open("/home/sofia96/Downloads/trajactory-learning/Sampletest/2013122701001.traj","w")
#get the(tno,state) dictionary
CarObs_Dict={}
Tno=0
for line in Traj:
	if line[0]=='t':
		Tno=int(line[4:])
		CarObs_Dict[Tno]=[]
	else:
		CarObs_Dict[Tno].append(line)

#transverse string to list
for key,val in CarObs_Dict.items():
	tmplist=[]
	for ele in val:
		strlist=ele.split(",")
		tmplist.append([float(i) for i in strlist])
	CarObs_Dict[key]=tmplist

#extract features that we are interested
for key,val in CarObs_Dict.items():
	tmpvalue=[]
	for item in val:
		yaw=math.atan(item[10]) #radius,x/y
		tmpvalue.append([item[0],item[8],item[9],yaw,item[12]])
	CarObs_Dict[key]=tmpvalue
#replace first ele with int
for key,val in CarObs_Dict.items():
	CarObs_Dict[key]=replace_float_to_int(val)

#normalize the dict elements to have the same length
CarObs_Dict=Traj_normalize(CarObs_Dict)

#get x distance mins 
CarObs_AvDis_Dict={}
for key,val in CarObs_Dict.items():
	valarray=np.array(val)
	thisDis=np.power(valarray[:,1],2)+np.power(valarray[:,2],2)
	AvDis=thisDis.mean()
	CarObs_AvDis_Dict[key]=AvDis

#delete elements not need
for index in range(0,const.CARTRUNC):
	del CarObs_AvDis_Dict[getminkey(CarObs_AvDis_Dict)]
for key in CarObs_AvDis_Dict:
	del CarObs_Dict[key]

show_traj_in_files("show.txt",CarObs_Dict)

#write to h5 file
Save_traj_to_h5(CarObs_Dict)

NormedTraj.close()
Traj.close()
