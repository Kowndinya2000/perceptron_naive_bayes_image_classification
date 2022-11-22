def get_data(dtype="training"):
    fp = open("digitdata/"+dtype+"images", "r")
    count = 0
    data_point = [[0 for _ in range(28)] for _ in range(28)]
    data = []
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
            data.append(data_point)
        else:
            line = a.replace("\n","")
            if "+" in line or "#" in line:
                c  = 0 
                for i in str(line):
                    if i == '+' or i == '#':
                        data_point[count-1][c] = 1
                    c += 1        
    fp.close()
    actual_y = []
    fp = open("digitdata/"+dtype+"labels","r")
    for a in fp:
        actual_y.append(int(a.replace("\n","")))
    fp.close()
    return data, actual_y