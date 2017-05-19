import numpy as np

#normilization data to be in [0,1]
def normalize(array,begin,end):
	for index in range(begin,end):
		arraymax=array[:,index].max()
		arraymin=array[:,index].min()
		for j in range(0,array.shape[0]):
			array[j][index]=(array[j][index]-arraymin)/(arraymax-arraymin)
	return array

#compare function
def first_ele(l):
	return l[0]
