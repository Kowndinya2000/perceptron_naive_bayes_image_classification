#############################
# Usage :  Python3 RFDigit.py -percentDataToBeUsedForTraining
#############################

from sklearn.ensemble import RandomForestClassifier
import random_forest_utils
import time
import numpy as np

dim = 28

if __name__ == "__main__":

    print("\n//////////////**STATS FOR RANDOM FOREST DIGITS**//////////////////")
    for training_percent in range(10,101,10):
        print("////⦿ Using", str(training_percent) + "%"+" of training data (RUNNING 50 SIMULATIONS)////")

        time_record = np.empty(50, dtype=float)
        prediction_errors = np.empty(50, dtype=float)

        for iter in range(1,51):
            start_time = time.time()
            data = random_forest_utils.readData('digitdata/trainingimages')
            labels = list(map(int, random_forest_utils.readData('digitdata/traininglabels')))

            partdata, partlabels = random_forest_utils.formatData(data, labels, training_percent, dim)

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(partdata, partlabels)

            time_record[iter-1] = time.time()-start_time

            testdata = random_forest_utils.readData('digitdata/testimages')
            testlabels = list(map(int, random_forest_utils.readData('digitdata/testlabels')))

            testPartData, testPartLabels = random_forest_utils.formatData(
                testdata, testlabels, 100, dim)
            accuracy = rf.score(testPartData, testPartLabels)
            prediction_errors[iter-1] = 1-accuracy
        print("// ➔ Average Training Time:", str(round(np.mean(time_record),2))+"sec")
        print("// ➔ Average Prediction Error:", str(round(100*np.mean(prediction_errors),2))+"%")
        print("// ➔ Standard Deviation:",np.std(prediction_errors))    
    print("//////////////**END OF STATS FOR RANDOM FOREST DIGITS**//////////////////")