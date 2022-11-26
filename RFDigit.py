#############################
#USage :  Python3 RFDigit.py -percentDataToBeUsedForTraining
#############################

import sys
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import RFUtils

# print(sys.path)
dim=28

if __name__ == '__main__':
    DataPercent=int(sys.argv[1])   
        
    data=RFUtils.readData('digitdata/trainingimages')
    labels=list(map(int,RFUtils.readData('digitdata/traininglabels')))
    # print(type(labels[0]))    
    
    partdata,partlabels=RFUtils.formatData(data,labels,DataPercent,dim)    
    
    print('Training Started...')
    rf=RandomForestClassifier(n_estimators=100)
    rf.fit(partdata,partlabels)
    
    print('Training Complete...')
    
    print('Testing...')

    
    testdata=RFUtils.readData('digitdata/testimages')
    testlabels=list(map(int,RFUtils.readData('digitdata/testlabels')))
    # print(type(labels[0]))
    
    
    testPartData,testPartLabels=RFUtils.formatData(testdata,testlabels,100,dim)          
   
    pred=rf.predict(testPartData)
    print ("Classification Report")
    print(classification_report(testPartLabels, pred))
    print ("Confusion Report")
    print(confusion_matrix(testPartLabels, pred))
    
   