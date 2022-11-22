fp = open("digitdata/trainingimages", "r")
count = 0
data_point = [[0 for _ in range(28)] for _ in range(28)]
training_data = []
for a in fp:
    count += 1
    if count == 1:
        data_point = [[0 for _ in range(28)] for _ in range(28)]
        line = a.replace("\n","")
        if "+" in line or "#" in line:
            c  = 0
            for i in str(line):
                if i == '+' or i == '#':
                    data_point[count-1][c] = 1
                c += 1
    elif count == 28: 
        line = a.replace("\n","")
        if "+" in line or "#" in line:
            c  = 0 
            for i in str(line):
                if i == '+' or i == '#':
                    data_point[count-1][c] = 1
                c += 1
        count = 0
        training_data.append(data_point)
    else:
        line = a.replace("\n","")
        if "+" in line or "#" in line:
            c  = 0 
            for i in str(line):
                if i == '+' or i == '#':
                    data_point[count-1][c] = 1
                c += 1        
print(len(training_data))
print(training_data[0][:10])
fp.close()
fp2 = open('digits','a')
for dp in training_data:
    for r in dp:
        for c in r:
            if c == 1:
                fp2.write("#")
            else:
                fp2.write(" ")
        fp2.write("\n")
fp2.close()
