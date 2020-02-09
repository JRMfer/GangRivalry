import glob
import numpy as np
import math
import csv
import matplotlib.pyplot as plt


# load the observed rivalry matrix
observe = list(csv.reader(open('observed.csv')))

paths = ['BM', 'SLBN', 'GRAV']

# loop over results of the different walking methods
for path in paths:
	npy_vars = []


	# read the .npy files in npy files
	for file_name in glob.glob(path + '/*.npy'):
		npy_vars.append(np.load(file_name))

	rivalries = []

	# make the directed weighted rivalry matrices
	for dicts in npy_vars:
		rivalry = []
		for i in range(len(dicts)):
			row = []

			# find the total interactions for each gang
			total_interactions = sum(dicts[i])
			for j in range(len(dicts[i])):
				if total_interactions:
					row.append(dicts[i][j] / total_interactions)
				else:
					row.append(0)
			rivalry.append(row)
		rivalries.append(rivalry)

	# list with the thresholds to test
	thresholds = [i * 0.001 for i in range(1,400)]

	all_rivalries = []
	for threshold in thresholds:
		all_matrix = []
		
		# create the undirected unweighted matrixes per threshold value
		for matrix in rivalries:
			
			matrix_to_threshold = np.zeros((len(matrix), len(matrix)))	
			
			for i in range(len(matrix)):
				for j in range(len(matrix[i])):
					
					# if either of the rivalry matrix surpases the threshold, add an edge
					if matrix[i][j] > threshold:
						matrix_to_threshold[i][j] = 1
						matrix_to_threshold[j][i] = 1
			all_matrix.append(matrix_to_threshold)

		all_rivalries.append(all_matrix)


	all_measures = []
	
	# loop over all the unweighted rivalry matrices at every threshold
	for thres in all_rivalries:
		measures = {}
		measures['ACC'] = []
		measures['F1'] = []
		measures['MCC'] = []
		measures['DENSITY'] = []
		for matrix in thres:
			TP = 0
			FP = 0
			TN = 0
			FN = 0
			degree = []
			
			# check how much the observed matrix and the rivalry matrices are similar
			for i in range(len(matrix)):
				degree.append(np.sum(matrix[i,:]))
				for j in range(len(matrix[i])):
					if matrix[i][j]:
						if int(observe[i][j][-1]):
							TP += 1
						else:
							FP += 1
					else:
						if int(observe[i][j][-1]):
							FN += 1
						else:
							TN += 1

			# divide by 2, because each edge is counted double
			TP = TP/2
			FP = FP/2
			TN = TN/2
			FN = FN/2

			# calculate the ACC, F1, MCC and DENSITY
			measures['ACC'].append((TP + TN)/(TP + TN + FN + FP))
			measures['F1'].append((2 * TP)/(2 * TP + FP + FN))
			measures['MCC'].append((TN * TP - FN * FP) / (math.sqrt((TP + FP) 
									* (TP + FN) * (TN + FN) * (TN + FP))))
			measures['DENSITY'] = np.sum(degree)/(len(matrix) * (len(matrix)-1))

		all_measures.append(measures)



	# ACC
	plt.figure()

	means_ACC = []
	std_ACC = []

	for thr in all_measures:
		means_ACC.append(np.mean(thr['ACC']))
		std_ACC.append(np.std(thr['ACC']))

	plt.plot(thresholds, means_ACC, color = 'blue')
	plt.errorbar(thresholds, means_ACC, yerr = std_ACC, color = 'blue', alpha = 0.3)
	plt.xlabel('Threshold (T)', fontsize = 15)
	plt.ylabel('Accuracy (ACC)', fontsize = 15)

	plt.savefig('ACC_' + path + '.png')
	plt.figure()
	means_F1 = []
	std_F1 = []

	for thr in all_measures:
		means_F1.append(np.mean(thr['F1']))
		std_F1.append(np.std(thr['F1']))

	plt.plot(thresholds, means_F1, color = 'blue')
	plt.errorbar(thresholds, means_F1, yerr = std_F1, color = 'blue', alpha = 0.3)
	plt.xlabel('Threshold (T)', fontsize = 15)
	plt.ylabel('Accuracy (F1)', fontsize = 15)
	plt.savefig('F1_'+ path + '.png')


	plt.figure()
	means_MCC = []
	std_MCC = []

	for thr in all_measures:
		means_MCC.append(np.mean(thr['MCC']))
		std_MCC.append(np.std(thr['MCC']))
		print()

	plt.plot(thresholds, means_MCC, color = 'blue')
	plt.errorbar(
		thresholds, means_MCC, yerr = std_MCC, 
		color = 'blue', alpha = 0.3
		)
	plt.xlabel('Threshold (T)', fontsize = 15)
	plt.ylabel('Accuracy (MCC)', fontsize = 15)

	plt.savefig('MCC_'+ path + '.png')