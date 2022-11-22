import copy
import random 
from util_digits import *
import sys

def getLabel(ino, dtype="test"):
    global weights
    global validation_data
    global test_data
    max_label = 0
    max_f = weights[0][0]
    if dtype == "validation":
        for x in range(len(validation_data[ino])):
            for y in range(len(validation_data[ino][x])):
                max_f += weights[0][1][x][y]*validation_data[ino][x][y] 

        for l in range(1,10):
            f = weights[l][0]
            for x in range(len(validation_data[ino])):
                for y in range(len(validation_data[ino][x])):
                    f += weights[l][1][x][y]*validation_data[ino][x][y]
            if max_f < f:
                max_label = l
                max_f = f 
        return max_label
    for x in range(len(test_data[ino])):
        for y in range(len(test_data[ino][x])):
            max_f += weights[0][1][x][y]*test_data[ino][x][y] 
    
    for l in range(1,10):
        f = weights[l][0]
        for x in range(len(test_data[ino])):
            for y in range(len(test_data[ino][x])):
                f += weights[l][1][x][y]*test_data[ino][x][y]
        if max_f < f:
            max_label = l
            max_f = f 
    return max_label    

def update(label, ino):
    global weights
    global features
    max_label = None
    max_f = weights[label][0]

    for x in range(len(features[ino])):
        for y in range(len(features[ino][x])):
            max_f += weights[label][1][x][y]*features[ino][x][y] 
    
    for l in range(10):
        if l != label:
            f = weights[l][0]
            for x in range(len(features[ino])):
                for y in range(len(features[ino][x])):
                    f += weights[l][1][x][y]*features[ino][x][y]
            if max_f < f:
                max_label = l
                max_f = f 
    if max_label is not None and max_label != label:
        weights[label][0] += 1
        weights[max_label][0] -= 1
        for i in range(len(weights[label][1])):
            for j in range(len(weights[label][1][i])):
                weights[label][1][i][j] += features[ino][i][j]
                weights[max_label][1][i][j] -= features[ino][i][j]

def get_accuracy(dtype="test"):
    global validation_data
    global test_data
    global validation_y
    global test_y
    accuracy = 0
    if dtype == "validation":
        for ino in range(len(validation_data)):
            predicted_label = getLabel(ino,dtype="validation")
            if predicted_label == validation_y[ino]:
                accuracy += 1
        accuracy = accuracy/len(validation_data)  
        return accuracy 
    for ino in range(len(test_data)):
        predicted_label = getLabel(ino)
        if predicted_label == test_y[ino]:
            accuracy += 1
    accuracy = accuracy/len(test_data)
    return accuracy
          
if __name__ == "__main__":
    
    training_percent = int(sys.argv[1])
    
    print("\n////////////////////////////////")
    print("// ⦿ training data used:", str(training_percent)+ "%")
    
    training_data, training_y = get_data()    
    validation_data, validation_y = get_data(dtype="validation")
    test_data, test_y = get_data(dtype="test")

    features = copy.deepcopy(training_data)
    max_iterations = 3
    weights = [[random.uniform(-1,1), [[random.uniform(-1,1) for _ in range(28)] for _ in range(28)]] for _ in range(10)]
    
    inos = [x for x in range(50*training_percent)]
    while(max_iterations > 0):
        for label, ino in zip(training_y[:len(inos)], inos):
            update(label, ino)
        max_iterations -= 1

    validation_accuracy = get_accuracy(dtype="validation")
    print("// ➔ Validation Accuracy:", str(round(validation_accuracy*100))+"%")

    test_accuracy = get_accuracy()
    print("// ➔ Test Accuracy:", str(round(test_accuracy*100, 2))+"%")
    print("////////////////////////////////")