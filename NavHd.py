import numpy as np
import h5py
import opt

#time,yaw,x,y,speed
#Save nav file to h5file
def Save_nav_to_h5(array):
	f=h5py.File("/home/sofia96/Downloads/trajactory-learning/h5/2013122701.h5","r+")
	#nav=f['.NAV']#
	thisNav=f.create_group(".NAV/001")
	thisNav=f['.NAV/001']
	t0=thisNav.create_dataset("ego",data=opt.normalize(array,1,array.shape[1]))
	f.close()

def Nav_normalize(filename,trajtimestamp,name):
	Nav=open(filename,"r")

#load nav file to the list
	ListofFile=[]
	for line in Nav:
		ListofFile.append(line.split('\t'))
	EgoCarNavList=[]
	for ele in ListofFile:
		EgoCarNavList.append([float(i) for i in ele])

#extract items needed
	EgoCarNavArray=np.array(EgoCarNavList)
	EgoCarArray=np.vstack((EgoCarNavArray[:,0],EgoCarNavArray[:,3],EgoCarNavArray[:,4]%6.28,EgoCarNavArray[:,5],EgoCarNavArray[:,7]))
	EgoCarArray=EgoCarArray.transpose()
	EgoCarList=[]
	for index in range(0,EgoCarArray.shape[0]):
		EgoCarList.append(list(EgoCarArray[index]))
	EgoCarTimeStamp=list(int(i) for i in list(EgoCarArray[:,0]))

#truncate nav array out of traj's timestamp
	num=0
	EgoCarObsList=[]
	for ele in EgoCarList:
		if int((ele[0])+2) in trajtimestamp:
			ele[0]=ele[0]-2
			EgoCarObsList.append(ele)
			num=num+1
		elif int(ele[0]) in trajtimestamp:
			EgoCarObsList.append(ele)
			num=num+1
		elif (int(ele[0])+1) in trajtimestamp:
			ele[0]=ele[0]-1
			EgoCarObsList.append(ele)
			num=num+1
	print(num)

	#resort EgocarObsList
	SortedEgoCarObsList=sorted(EgoCarObsList,key=opt.first_ele)

	#normalize positions
	Globx=SortedEgoCarObsList[0][2]
	Globy=SortedEgoCarObsList[0][3]
	SortedEgoCarArray=np.array(SortedEgoCarObsList)
	SortedEgoCarArray[:,2]=SortedEgoCarArray[:,2]-Globx
	SortedEgoCarArray[:,3]=SortedEgoCarArray[:,3]-Globy


	#save to h5files
	Save_nav_to_h5(SortedEgoCarArray)

	Nav.close()



