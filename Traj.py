import sys
import os
import numpy as np
import h5py 
sys.path.append('/home/sofia96/Downloads/trajactory-learning/script')

#define several const integers
from Myclass import const
const.MAXEPX=10
const.MAXEPY=20
const.CARTRUNC=5

def getminkey(d):
	v=list(d.values())
	k=list(d.keys())
	return k[v.index(min(v))]

#normalize the dict elements to have the same length
def Traj_normalize(dic):
	#extract all the timestamps
	Timestamp=set()
	TimestampDic={}
	for key,val in dic.items():
		valarray=np.array(val)
		Timestamp=Timestamp.union(set(valarray[:,0]))
		TimestampDic[key]=set(valarray[:,0])
	#unify the timestamps
	for key,val in dic.items():
		velocity=val[0][5]
		for ele in Timestamp:
			if ele not in TimestampDic[key]:
				dic[key].append([ele,const.MAXEPX,const.MAXEPY,0,0,velocity])

	return dict(sorted(dic.items(),key=lambda d:d[1],reverse=True))

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
		tmpvalue.append([item[0],item[8],item[9],item[10],item[11],item[12]])
	CarObs_Dict[key]=tmpvalue

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

#write to h5 file
Save_traj_to_h5(CarObs_Dict)

NormedTraj.close()
Traj.close()
