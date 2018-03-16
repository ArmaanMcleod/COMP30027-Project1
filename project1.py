import numpy as np

from csv import reader
from pprint import pprint
from collections import defaultdict
from itertools import chain
from operator import itemgetter

# This function should open a data file in csv, and transform it into a usable format
def preprocess(filename):

    # all lines in file go here
    lines = []

    # open file
    with open(filename) as infile:

        # convert to reader object
        csv_reader = reader(infile)

        # loop over each line and add it to resultOnant list
        for line in csv_reader:
            lines.append(line)

    return lines

# This function should build a supervised NB model
def train_supervised(training_data):
    priors = defaultdict(int)

    # count prior probabilities
    for line in training_data:
        col = line[-1]
        priors[col] += 1 / len(training_data)

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

    for class_name in posteriers:
        product = 1

        # go over attributes of the trained data and instances at once
        # get the hashed probability
        for attribute, value in zip(posteriers[class_name], instance):
            prob = posteriers[class_name][attribute][value]

            # if probability is non-zero, accumulate it
            # otherwise, accumulate the smoothing factor
            if prob:
                product *= prob
            else:
                product *= EPSILON

        class_max[class_name] = product * priors[class_name]

    # return a tuple of the class and maximal probability
    return max(class_max.items(), key=itemgetter(1))

# This function should evaluate a set of predictions, in a supervised context
def evaluate_supervised(priors, posteriers, data):
    # keep a counter of correct instances found
    correct = 0

    # go over each instance in data set
    # get predicted class for each instance
    for instance in data:
        predict_class, _ = predict_supervised(priors, posteriers, instance)

        # if class is identical to the instances last column
        # increment the count
        if predict_class == instance[-1]:
            correct += 1

    return correct/len(data)

# This function should build an unsupervised NB model
def train_unsupervised(training_data):
    # get sorted list of classes in data set
    classes = sorted(set(map(itemgetter(-1), training_data)))

    # instances with attributes and distributions
    labelled_instances = []

    # priors of distribution counts
    priors = defaultdict(float)

    # calculate priors of random distribtions
    # and transform training data into randomized labelled data
    for instance in training_data:

        # get random distribution of classes
        rand_distribution = np.random.dirichlet(np.ones(len(classes)), size=1)
        
        # create tupled pairs of (class, distribution)
        class_distribution = list(zip(classes, *rand_distribution))

        # count priors
        for class_name, dist in class_distribution:
            priors[class_name] += dist / len(training_data)

        # add attributes to a dictionary, excluding class column
        row = dict(enumerate(instance[:-1]))

        # add distrubution dictionary 
        row['distribution'] = dict(class_distribution)

        labelled_instances.append(row)

    pprint(labelled_instances)

    #for instance in labelled_instances:


    #print(priors)

    


# This function should predict the class distribution for a set of instances, based on a trained model
def predict_unsupervised():
    return

# This function should evaluate a set of predictions, in an unsupervised manner
def evaluate_unsupervised():
    return



def main():
    datasets = ['breast-cancer.csv',
                'car.csv',
                'hypothyroid.csv',
                'mushroom.csv']

    #for file in datasets:
        #data = preprocess(file)
        #trained_data = train_supervised(data)
        #print(trained_data)

    # testing on just one dataset
    data = preprocess(datasets[1])

    # SUPERVISED HERE
    priors, posteriers = train_supervised(data)

    # test row here
    instance = ['vhigh','vhigh', '2', '4', 'small', 'med', 'unnac']
    predict = predict_supervised(priors, posteriers, instance[:-1])

    evaluate = evaluate_supervised(priors, posteriers, data)

    print(evaluate)

    # UNSUPERVISED HERE
    print(train_unsupervised(data))


    #print(trained_data)


if __name__ == '__main__':
    main()
