import random
from collections import Counter

def readData(path):
    print('Reading file ',path,'...')
    with open(path) as f:
        data=f.read().splitlines()
    print('Reading Complete.',len(data), ' lines read.')
    return data
    
    

def formatData(data,labels,DataPercent,dim):
    formatedData=[]
    i=0
    while i< len(data):
        img=[]
       
        line=data[i:i+dim]
        img=[]
        for l in line:
            a=[]
            for k in l:
                if k== ' ':
                    a.append(0)
                else:
                    a.append(1)
            img.extend(a)

        formatedData.append(img)
            # print(i)
        i+=dim
        


    temp=list(zip(formatedData,labels))
    random.shuffle(temp)
    formatedData,labels=zip(*temp)
    formatedData,labels=list(formatedData),list(labels)
    i=0
    Digitscounts=Counter(labels)


    count=[0]*len(Digitscounts.keys())
    partdata=[]
    partlabels=[]
    for i,x in enumerate(labels):
        # print(i,x)
            if count[x]<=(DataPercent/100)*Digitscounts[x]:
                count[x]+=1
                partdata.append(formatedData[i])
                partlabels.append(labels[i])
    return partdata,partlabels
