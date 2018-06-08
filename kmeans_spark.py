from __future__ import print_function

import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
import numpy as np
import pandas as pd
import random
import math
import time

def load_feature(filename):
	df = pd.read_csv(filename)
	df = df.drop(columns = ['class'])
	return df


def load_label(filename):
	df = pd.read_csv(filename)
	df = df[['class']]
	return df


def group(node,center):
    min_dist = 9999999
    group_idx = 0
    for i in range(len(center)):
        dist = np.sum(pow((node - center[i]),2))
        if dist < min_dist:
            min_dist = dist
            group_idx = i
    return group_idx


def kmeans(RDDdf,K,old_center,finished):
	group_result = RDDdf.map(lambda n: (group(n,old_center), (n,1) ) )
	# print(group_result.take(10))
	sum_node = group_result.reduceByKey(lambda x,y: x+y)
	# print(sum_node.take(10))
	new_point = sum_node.map(lambda m: (m[0], m[1][0]/m[1][1] ) )
	print(new_point.take(10))
	# print("===============================")
	new_point_list = new_point.map(lambda m:(m[1])).collect()
	# print(type(new_point))
	# print("===============================")
	if(finished == True):
		return group_result
	else:
		return new_point_list

def sse(RDDdf,center,num_cluster):
	dist = np.empty([num_cluster])
	# print(type(dist[2]))
	group = RDDdf.collect()
	# print(group[0][1])   #(array([5.1, 3.5, 1.4, 0.2]), 1)
	# print(group[0][1][0])    #[5.1 3.5 1.4 0.2]
	# print(group[0][1][0][1])	#3.5
	# print(type(center[1][2]))
	# print(pow(group[0][1][0][0] - center[1][0],2))
	for i in range(num_cluster):
		try:
			dist[i] = dist[i] + float(pow(group[0][i][0][0] - center[i][0],2) + pow(group[0][i][0][1] - center[i][1],2) + pow(group[0][i][0][2] - center[i][2],2) + pow(group[0][i][0][3] - center[i][3],2) + pow(group[0][i][0][4] - center[i][4],2) + pow(group[0][i][0][5] - center[i][5],2))
		except TypeError:
			pass
		except IndexError:
			pass
	total_dist = sum(dist)
	return total_dist


def main():
	'''initial spark'''
	conf = SparkConf().setMaster("local").setAppName("spark_learn").set("spark.ui.port", "5068")
	sc = SparkContext(conf = conf)
	sqlContext = SQLContext(sc)

	'''initial center'''
	feature = load_feature("c20d6n1200000t.csv")
	# print(feature.head())
	label = load_label("c20d6n1200000t.csv")
	# print(feature.head())
	# print(label.head())
	RDDdf = sc.parallelize(feature.values)
	# print(RDDdf.take(5))
	
	K = 20

	center = RDDdf.takeSample(False,K,1)
	print("===============================")

	'''run kmeans algo'''
	i = 0
	while i < 10:
		new_center = kmeans(RDDdf,K,center,False)
		# print(new_center)
		print("====================")
		center = new_center
		i = i + 1
	out = kmeans(RDDdf,K,center,True)
	# print(type(center))
	# print(out.take(1000))
	print("sse = ",sse(out,center,K))


	
if __name__ == '__main__':
	start_time = time.time()
	main()
	end = time.time()
	print("Execution time :",( end - start_time))