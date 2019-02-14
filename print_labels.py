import csv
import sys

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("error.  specify file path")
		exit()
	csv_path = sys.argv[1]
	with open(csv_path, 'rt', encoding="utf8") as csvin:
	    reader = csv.reader(csvin)
	    true_labels = list(reader)
	    true_labels.pop(0) #get rid of header
	    #true_labels = true_labels[start:]
	counter = 0
	for i in true_labels:
		print(i[1])
		if i[1] == 1 or i[1] == '1':
			counter += 1
	print("counter: {}".format(counter))