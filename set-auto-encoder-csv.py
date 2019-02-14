import pandas as pd
import sys

if __name__ == "__main__":

	if sys.argv[1] == "-h":
		print("\t - This program takes in NetExtractor CSV and outputs CSV with Features in order of which autoencoder they're assigned to")
		print("\t - Enter infile and outfile path")
		print("\t\texample: python3 set-auto-encoder-csv.py mirai3.csv mirai3_autoencoder.csv")
		exit()
	if len(sys.argv) != 3:
		print("Error, need 2 arguments")
		print("need: infile and outfile")
		exit()
	infile = sys.argv[1]
	outfile = sys.argv[2]
	df = pd.read_csv(infile, header=None)

	df = df[[67, 70, 99, 85, 92, 100, 86, 93, 107, 114, 106, 113,
	75, 72, 66, 69, 73, 76, 78, 79, 50, 57, 64, 36, 43, 49, 35, 42,
	46, 32, 39, 48, 34, 41, 13, 28, 10, 25, 62, 55, 53, 60, 56, 63,
	8, 23, 2, 17, 5, 20, 97, 83, 90, 104, 111, 54, 61, 47, 33, 40,
	52, 59, 45, 31, 38, 95, 81, 88, 102, 109, 7, 22, 4, 19, 1, 16,
	103, 105, 110, 112, 96, 98, 84, 91, 82, 89, 14, 29, 11, 26,
	58, 77, 51, 74, 44, 71, 37, 68, 30, 65, 12, 27, 9, 24, 6, 21, 
	0, 15, 3, 18, 101, 108, 94, 80, 87]]

	df.to_csv(outfile, header=None, index=False)

"""
Autoencoder 	Feature
1				[67, 70]
2 				[99, 85, 92, 100, 86, 93, 107, 114, 106, 113]
3				[75, 72, 66, 69, 73, 76, 78, 79]
4				[50, 57, 64, 36, 43]
6				[49, 35, 42]
7				[46, 32, 39, 48, 34, 41]
8				[13, 28, 10, 25, 62, 55, 53, 60, 56, 63]
9				[8, 23, 2, 17, 5, 20]
10				[97, 83, 90, 104, 111, 54, 61, 47, 33, 40]
11				[52, 59, 45, 31, 38, 95, 81, 88, 102, 109]
12				[7, 22, 4, 19, 1, 16]
13				[103, 105, 110, 112, 96, 98, 84, 91, 82, 89]
14				[14, 29, 11, 26], [58, 77, 51, 74]
15				[44, 71, 37, 68, 30, 65]
16				[12, 27, 9, 24, 6, 21, 0, 15, 3, 18]
17				[101, 108, 94, 80, 87]
"""