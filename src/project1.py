import csv
from collections import defaultdict
from pprint import pprint

# This function should open a data file in csv, and transform it into a usable format 
def preprocess(filename):

    # all lines in file go here
    lines = []

    # open file
    with open(filename, 'r') as infile:

        # convert to reader object
        reader = csv.reader(infile)

        # loop over each line and add it to resultant list
        for line in reader:
            lines.append(line)

    return lines

# This function should build a supervised NB model
def train_supervised(training_data):

    # posterier probablities
    priors = defaultdict(int)

    # number of lines in preprocessed training data
    num_lines = len(training_data)

    # count prior probabilities
    for line in training_data:
        col = line[-1]
        priors[col] += 1

    priors = {k: v / num_lines for k, v in priors.items()}

    posterier = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    for line in training_data:
        attributes = line[:-1]
        class_name = line[-1]
        for attribute, value in enumerate(attributes):
            posterier[class_name][attribute][value] += 1
 
    for class_name in posterier:
        for attribute in posterier[class_name]:
            sums = sum(posterier[class_name][attribute].values())
            for count in posterier[class_name][attribute]:
                posterier[class_name][attribute][count] /= sums

    return priors, posterier

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

    #print(data)

    print(train_supervised(data))


