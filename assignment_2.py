#assignment_1

import numpy
import csv
import random
import pickle

def norm(f):
    return f/255

def activation(x):
    return 1 if x >= 0 else 0

v_activation = numpy.vectorize(activation)

data = []
test = []

#Hyperparameters. Change these to complete assignment. 
epochs = 50
#l_r = 0.001

try:
    data = pickle.load(open("train.pickle", "rb"))
except(OSError, IOError) as _:
    with open("mnist_train.csv", 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            l = int(row.pop(0))
            row = [float(x)/255 for x in row]
            row.insert(0, l)
            data.append(row) 
        pickle.dump(data, open("train.pickle", "wb"))
        print("no train.pickle")
print("data read")

#Same as the try/except block above.
try: 
    test = pickle.load(open("test.pickle", "rb"))
except(OSError, IOError) as _:
    with open("mnist_test.csv", 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            l = int(row.pop(0))
            row = [float(x)/255 for x in row]
            row.insert(0, l)
            test.append(row)
        pickle.dump(test, open("test.pickle", "wb"))
        print("no test.pickle")
print("test read")
        
data_labels = []
test_labels = []

random.shuffle(data)
print("data shuffled")

for row in data:
    data_labels.append(row[0])
    row[0] = 1
print("data labels fetched")
    
for row in test:
    test_labels.append(row[0])
    row[0] = 1
print("test labels fetched")

data = numpy.array(data)
print("data converted")
data_labels = numpy.array(data_labels)
print("data labels converted")
test = numpy.array(test)
print("test converted")
test_labels = numpy.array(test_labels)
print("test labels converted")