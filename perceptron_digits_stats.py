# Task: Perceptron Classifier to classify 0-9 digits 
#  ///////  
#     ///
#    ///   
#   ///
'''
////////////////////////////////////////////////////////////////////////////////////////////////
// @author Kowndinya Boyalakuntla - <RUID 219002814> - <NetID kb1204>
// @usage: python3 perceptron_digits_stats.py  
///////////////////////////////////////////////////////////////////////////////////////////////
'''
import copy
from util_digits import *
import numpy as np
import time

## Predicts the class label on the given input image
## @Input: ino is ImageID and dtype is Dataset Type (validation/test)
## @Output: label (0-9)
def predict_label(ino, dtype="test"):
    global weights
    global validation_data
    global test_data
    max_label = 0
    max_f = weights[0][0]
    if dtype == "validation":
        max_f = np.sum(weights[0][1]*validation_data[ino])
        for l in range(1, 10):
            f = weights[l][0] + np.sum(weights[l][1]*validation_data[ino])
            if max_f < f:
                max_label = l
                max_f = f
        return max_label
    max_f = np.sum(weights[0][1]*test_data[ino])

    for l in range(1, 10):
        f = weights[l][0] + np.sum(weights[l][1]*test_data[ino])

        if max_f < f:
            max_label = l
            max_f = f
    return max_label

## Fine-tunes the weight matrix during the course of training process
## @Input: label is the class label of the input image, ino is its ImageID
def update_weights(label, ino):
    global weights
    global features
    max_label = None
    max_f = weights[label][0] + np.sum(weights[label][1]*features[ino])

    for l in range(10):
        if l != label:
            f = weights[l][0] + np.sum(weights[l][1]*features[ino])
            if max_f < f:
                max_label = l
                max_f = f
    if max_label is not None and max_label != label:
        weights[label][0] += 1
        weights[max_label][0] -= 1
        weights[label][1] += features[ino]
        weights[max_label][1] -= features[ino]

## Calculates the accuracy of the perceptron classifier
## @Input: dtype - Dataset type (test/validation)
## @Output: accuracy of the classifier on the test/validation dataset
def get_accuracy(dtype="test"):
    global validation_data
    global test_data
    global validation_y
    global test_y
    accuracy = 0
    if dtype == "validation":
        for ino in range(len(validation_data)):
            predicted_label = predict_label(ino, dtype="validation")
            if predicted_label == validation_y[ino]:
                accuracy += 1
        accuracy = accuracy/len(validation_data)
        return accuracy
    for ino in range(len(test_data)):
        predicted_label = predict_label(ino)
        if predicted_label == test_y[ino]:
            accuracy += 1
    accuracy = accuracy/len(test_data)
    return accuracy

## Main Function
## Training & Test data generation
## Feature Extraction - Each Pixel is a feature - Essentially the training_data is the feature matrix
## Training the classifier
## Prediction on the Test and Validation datasets 
## Printing the corresponding training time (sec) and prediction error (%) 
if __name__ == "__main__":
    print("\n//////////////**STATS FOR PERCEPTRON DIGITS**//////////////////")
    for training_percent in range(10,101,10):
        print("////⦿ Using", str(training_percent) + "%"+" of training data (RUNNING 50 SIMULATIONS)////")

        time_record = np.empty(50, dtype=float)
        prediction_errors = np.empty(50, dtype=float)

        for iter in range(1,51):
            start_time = time.time()
            training_data, training_y = get_data()

            ## Feature Extraction
            features = copy.deepcopy(training_data)
            max_iterations = 3
            weights = [[np.random.randn(1), np.random.randn(28,28)] for _ in range(10)]
            inos = [x for x in range(50*training_percent)]

            ## Perceptron classifier's training happens here
            while (max_iterations > 0):
                for label, ino in zip(training_y[:len(inos)], inos):
                    update_weights(label, ino)
                max_iterations -= 1

            time_record[iter-1] = time.time()-start_time
            test_data, test_y = get_data(dtype="test")
            test_accuracy = get_accuracy()
            prediction_errors[iter-1] = 1-test_accuracy
        
        print("// ➔ Average Training Time:", str(round(np.mean(time_record),2))+"sec")
        print("// ➔ Average Prediction Error:", str(round(100*np.mean(prediction_errors),2))+"%")
        print("// ➔ Standard Deviation:",np.std(prediction_errors))    
    print("//////////////**END OF STATS FOR PERCEPTRON DIGITS**//////////////////")
    