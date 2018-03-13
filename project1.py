import csv
from pprint import pprint
from collections import defaultdict
from operator import itemgetter

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

    # posteriers probablities
    priors = defaultdict(int)

    # number of lines in preprocessed training data
    num_lines = len(training_data)

    # count prior probabilities
    for line in training_data:
        col = line[-1]
        priors[col] += 1

    # update dictionary with probabilites instead of counts
    priors = {k: v / num_lines for k, v in priors.items()}

    # create posteriers data structure with triple nested defaultdicts
    # perhaps a more simplified data structure could be used here
    # initialising int here initially for storing frequencies
    # initialise frequency to 1 for smoothing
    posteriers = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

    # Loop over each line in preprocessed training data
    for line in training_data:

        # obtain class and attributes
        attributes = line[:-1]
        class_name = line[-1]

        # count each item
        # using indices as attribute headers since not given in dataset
        # indice starts at one
        # perhaps datasets could be modified to include header names
        for attribute, freq in enumerate(attributes):
            posteriers[class_name][attribute][freq] += 1
 
    # transform posteriers counts into probabilities
    for class_name in posteriers:
        for attribute in posteriers[class_name]:

            # sum counts over each dict
            # This will be the same accross each class
            sums = sum(posteriers[class_name][attribute].values())

            # update to probabilites by dividing freq/sums
            for freq in posteriers[class_name][attribute]:
                posteriers[class_name][attribute][freq] /= sums

    # return a tuple of the two above data structures
    return priors, posteriers

# This function should predict the class for a set of instances, based on a trained model 
def predict_supervised(priors, posteriers, instance):

    # epsilon value for smoothing
    EPSILON = 0.000000000001

    # dictionary holding maximal for each class
    class_max = {}

    # go through each class
    for class_name in posteriers:

        # product accumulator
        product = 1

        # go over attributes of the trained data and instances at once
        for attribute, value in zip(posteriers[class_name], instance):

            # get the hashed probability
            loc = posteriers[class_name][attribute][value]

            # if probability is non-zero, accumulate it
            if loc:
                product *= loc

            # otherwise, accumulate the smoothing factor
            else:
                product *= EPSILON

        # add accumulated values to class maximals
        class_max[class_name] = product * priors[class_name]

    # return a tuple of the class and maximal probability
    return max(class_max.items(), key=itemgetter(1))

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

def main():
    datasets = ['breast-cancer-dos.csv',
                'car-dos.csv',
                'hypothyroid-dos.csv',
                'mushroom-dos.csv']

    #for file in datasets:
        #data = preprocess(file)
        #trained_data = train_supervised(data)
        #print(trained_data)

    # testing on just one dataset
    data = preprocess(datasets[1])
    priors, posteriers = train_supervised(data)

    # test row here
    instance = ['vhigh','vhigh', '2', '4', 'small', 'med', 'unnac']
    print(predict_supervised(priors, posteriers, instance[:-1]))


    #print(trained_data)


if __name__ == '__main__':
    main()
