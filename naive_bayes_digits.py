from util_digits import *
import sys

if __name__ == "__main__":

    training_percent = int(sys.argv[1])
    print("\n////////////////////////////////")
    print("// ⦿ training data used:", str(training_percent) + "%")

    training_data, training_y = get_data()

    features = []

    inos = [x for x in range(50*training_percent)]
    prob_table_digits = [[[[0 for _ in range(17)] for _ in range(
        7)] for _ in range(7)] for _ in range(10)]
    num_digits = [0 for _ in range(10)]

    for ino in inos:
        feature = [[0 for _ in range(7)] for _ in range(7)]
        for x in range(7):
            for y in range(7):
                i = 4*x
                j = 4*y
                for a in range(i, i+4):
                    for b in range(j, j+4):
                        feature[x][y] += training_data[ino][a][b]
        features.append(feature)

    for ino in inos:
        num_digits[training_y[ino]] += 1
        for x in range(7):
            for y in range(7):
                prob_table_digits[training_y[ino]
                                  ][x][y][features[ino][x][y]] += 1

    test_accuracy = 0
    test_data, test_y = get_data(dtype="test")
    features_test = []
    for ino, label in enumerate(test_y):
        feature = [[0 for _ in range(7)] for _ in range(7)]
        for x in range(7):
            for y in range(7):
                i = 4*x
                j = 4*y
                for a in range(i, i+4):
                    for b in range(j, j+4):
                        feature[x][y] += test_data[ino][a][b]
        features_test.append(feature)

    for ino in range(len(test_data)):
        probs = [num_digits[i]/sum(num_digits) for i in range(10)]
        for x in range(7):
            for y in range(7):
                for i in range(10):
                    factor = prob_table_digits[i][x][y][features_test[ino][x][y]]/sum(
                        prob_table_digits[i][x][y]) if prob_table_digits[i][x][y][features_test[ino][x][y]] != 0 else 1/num_digits[i]
                    probs[i] *= factor
        max_label = 0
        max_prob = probs[0]
        for i in range(1, 10):
            if max_prob < probs[i]:
                max_prob = probs[i]
                max_label = i

        if max_label == test_y[ino]:
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
                i = 4*x
                j = 4*y
                for a in range(i, i+4):
                    for b in range(j, j+4):
                        feature[x][y] += validation_data[ino][a][b]
        features_validation.append(feature)

    for ino in range(len(validation_data)):
        probs = [num_digits[i]/sum(num_digits) for i in range(10)]
        for x in range(7):
            for y in range(7):
                for i in range(10):
                    factor = prob_table_digits[i][x][y][features_validation[ino][x][y]]/sum(
                        prob_table_digits[i][x][y]) if prob_table_digits[i][x][y][features_validation[ino][x][y]] != 0 else 1/num_digits[i]
                    probs[i] *= factor
        max_label = 0
        max_prob = probs[0]
        for i in range(1, 10):
            if max_prob < probs[i]:
                max_prob = probs[i]
                max_label = i

        if max_label == validation_y[ino]:
            validation_accuracy += 1

    validation_accuracy /= len(validation_data)

    print("// ➔ validation Accuracy:", str(round(validation_accuracy*100))+"%")

    print("////////////////////////////////")
