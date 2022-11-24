from util_faces import *
import sys
          
if __name__ == "__main__":
    
    training_percent = int(sys.argv[1])
    print("\n////////////////////////////////")
    print("// ⦿ training data used:", str(training_percent)+ "%")
    
    training_data, training_y = get_data()    

    features = []
    
    inos = [x for x in range(451*training_percent//100)]
    prob_table_face = [[[0 for _ in range(101)] for _ in range(7)] for _ in range(7)]
    prob_table_not_face = [[[0 for _ in range(101)] for _ in range(7)] for _ in range(7)]
    num_faces = 0

    for ino in inos:
        feature = [[0 for _ in range(7)] for _ in range(7)]
        for x in range(7):
            for y in range(7):
                i = 10*x
                j = 10*y
                for a in range(i, i+10):
                    for b in range(j, j+10):
                        feature[x][y] += training_data[ino][a][b]
        features.append(feature)

    for ino in inos:    
        if training_y[ino] == 1:
            num_faces += 1
            for x in range(7):
                for y in range(7):
                    prob_table_face[x][y][features[ino][x][y]] += 1
        else:
            for x in range(7):
                for y in range(7):
                        prob_table_not_face[x][y][features[ino][x][y]] += 1

    test_accuracy = 0
    test_data, test_y = get_data(dtype="test")
    features_test = []
    for ino, label in enumerate(test_y):
        feature = [[0 for _ in range(7)] for _ in range(7)]
        for x in range(7):
            for y in range(7):
                i = 10*x
                j = 10*y
                for a in range(i, i+10):
                    for b in range(j, j+10):
                        feature[x][y] += test_data[ino][a][b]
        features_test.append(feature)

    for ino in range(len(test_data)):
        pface = num_faces/len(inos)
        pnotface = 1 - pface
        for x in range(7):
            for y in range(7):
                factor = prob_table_face[x][y][features_test[ino][x][y]]/sum(prob_table_face[x][y]) if prob_table_face[x][y][features_test[ino][x][y]] != 0 else 1/num_faces
                pface *= factor
                factor2 = prob_table_not_face[x][y][features_test[ino][x][y]]/sum(prob_table_not_face[x][y]) if prob_table_not_face[x][y][features_test[ino][x][y]] != 0 else 1/(len(inos)-num_faces)
                pnotface *= factor2
 
        if pface > pnotface and test_y[ino] == 1:
            test_accuracy += 1
        elif pface < pnotface and test_y[ino] == 0:
            test_accuracy += 1
           
    test_accuracy /= len(test_data)

    print("// ➔ Test Accuracy:", str(round(test_accuracy*100))+"%")
    
    validation_accuracy = 0
    validation_data, validation_y = get_data(dtype="validation")
    features_validation = []
    for ino, label in enumerate(validation_y):
        feature = [[0 for _ in range(7)] for _ in range(7)]
        for x in range(7):
            for y in range(7):
                i = 10*x
                j = 10*y
                for a in range(i, i+10):
                    for b in range(j, j+10):
                        feature[x][y] += validation_data[ino][a][b]
        features_validation.append(feature)

    for ino in range(len(validation_data)):
        pface = num_faces/len(inos)
        pnotface = 1 - pface
        for x in range(7):
            for y in range(7):
                factor = prob_table_face[x][y][features_validation[ino][x][y]]/sum(prob_table_face[x][y]) if prob_table_face[x][y][features_validation[ino][x][y]] != 0 else 1/num_faces
                pface *= factor
                factor2 = prob_table_not_face[x][y][features_validation[ino][x][y]]/sum(prob_table_not_face[x][y]) if prob_table_not_face[x][y][features_validation[ino][x][y]] != 0 else 1/(len(inos)-num_faces)
                pnotface *= factor2
 
        if pface > pnotface and validation_y[ino] == 1:
            validation_accuracy += 1
        elif pface < pnotface and validation_y[ino] == 0:
            validation_accuracy += 1
           
    validation_accuracy /= len(validation_data)
    
    print("// ➔ Validation Accuracy:", str(round(validation_accuracy*100))+"%")

    print("////////////////////////////////")