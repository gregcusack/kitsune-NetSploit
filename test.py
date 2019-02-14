
x = [1,2,3,4,5,6,7]

count = 0
for i in range(0, len(x)):
	print(x[i])
	x.append("a")
	count += 1
	if count == 100:
		break
print(x)

