import csv
from collections import defaultdict
from pprint import pprint

# This function should open a data file in csv, and transform it into a usable format 
def preprocess(filename):

    # all lines in file go here
    lines = []

    # open file
    with open(filename) as infile:

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

    # update dictionary with probabilites instead of counts
    priors = {k: v / num_lines for k, v in priors.items()}

    # create posterier data structure with triple nested defaultdicts
    # perhaps a more simplified data structure could be used here
    # initialising int here initially for storing frequencies
    posterier = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

    # Loop over each line in preprocessed training data
    for line in training_data:

        # obtain class and attributes
        attributes = line[:-1]
        class_name = line[-1]

        # count each item
        # using indices as attribute headers since not given in dataset
        # perhaps datasets could be modified to include header names
        for attribute, freq in enumerate(attributes):
            posterier[class_name][attribute][freq] += 1
 
    # transform posterier counts into probabilities
    for class_name in posterier:
        for attribute in posterier[class_name]:

            # sum counts over each dict
            # This will be the same accross each class
            sums = sum(posterier[class_name][attribute].values())

            # update to probabilites by dividing freq/sums
            for freq in posterier[class_name][attribute]:
                posterier[class_name][attribute][freq] /= sums

    # return a tuple of the two above data structures
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


