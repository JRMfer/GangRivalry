import csv
import glob
import numpy as np

for file in glob.glob('*.csv'):
	with open(file) as f:
		read = csv.reader(f, delimiter = ',')
		next(read)
		data = []
		for row in read:
			#print(row)
			data.append(row)

	data = data[-1][-1]

	data2 = []

	pos_min = 2
	pos_max = 13
	for j in range(29):
		row = []
		for i in range(29):
			row.append(data[pos_min: pos_max])
			pos_min = pos_max + 1
			pos_max += 12
			if not (i + 1) % 6:
				pos_min += 2
				pos_max += 2
		data2.append(row)
		pos_min += 3
		pos_max += 3

	print(data2)

	rivalry_data = []

	for c in data2:
		row = []
		for numb in c:
			row.append(int(float(numb[0:7]) * 10 ** int(numb[-1])))
		rivalry_data.append(row)

	print(rivalry_data)
	np.save(file[0:-4] + '.npy', rivalry_data)


