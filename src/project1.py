import csv
import numpy as np  
import pandas as pd
import csv
from collections import defaultdict

# This function should open a data file in csv, and transform it into a usable format 
def preprocess(filename):
    lines = []
    with open(filename, 'r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)

    return lines

# This function should build a supervised NB model
def train_supervised(training_data):
    class_counts = defaultdict(int)

    num_lines = len(training_data)

    for line in training_data:
        col = line[-1]
        class_counts[col] += 1

    priors = {k: v / num_lines for k, v in class_counts.items()}

    print(priors)

    posteriers = []
    for line in training_data:
        pass


# This function should predict the class for a set of instances, based on a trained model 
def predict_supervised():
    return

# This function should evaluate a set of predictions, in a supervised context 
def evaluate_supervised():
    return

# This function should build an unsupervised NB model 
def train_unsupervised():
    return

# This function should predict the class distribution for a set of instances, based on a trained model
def predict_unsupervised():
    return

# This function should evaluate a set of predictions, in an unsupervised manner
def evaluate_unsupervised():
    return

if __name__ == '__main__':
    dataset = 'C:\\Users\\Alex\\Documents\\mygit\\COMP30027-Project1\\datasets\\car-dos.csv'
    data = preprocess(dataset)

    print(data)

    print(train_supervised(data))


