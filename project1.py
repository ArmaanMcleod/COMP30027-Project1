import os
import numpy as np

from csv import reader
from pprint import pprint
from itertools import chain
from operator import itemgetter
from collections import Counter
from collections import defaultdict
from collections import OrderedDict

# This function should open a data file in csv
# transform it into a usable format
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

    # priors count dictionary
    priors = defaultdict(int)

    # count prior probabilities
    for line in training_data:
        col = line[-1]
        priors[col] += 1/len(training_data)

    # create posteriers data structure with triple nested defaultdicts
    # perhaps a more simplified data structure could be used here
    # initialising int here initially for storing frequencies
    # initialise frequency to 1 for smoothing
    posteriers = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

    # Loop over each line in preprocessed training data
    for line in training_data:

        # obtain class and attributes
        attributes, class_name = line[:-1], line[-1]

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

# This function should predict the class for a set of instances
# based on a trained model
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
def evaluate_supervised(priors, posteriers, training_data):

    # keep a counter of correct instances found
    correct = 0

    # go over each instance in data set
    # get predicted class for each instance
    for instance in training_data:
        predict_class, _ = predict_supervised(priors, posteriers, instance)

        # if class is identical to the instances last column
        # increment the count
        if predict_class == instance[-1]:
            correct += 1

    return correct/len(training_data)

# This function uses the cross validation strategy
# Which runs on a supervised NB model
def cross_validation(dataset, k):

    # divide dataset into k length partitions
    partitions = [part.tolist() for part in np.array_split(dataset, k)]

    # helper function for flattening a list
    flatten = lambda lst : list(chain.from_iterable(lst))

    # accuracy counter
    sums = 0

    for i, test_data in enumerate(partitions):

        # get every other partition except current test data
        training_data = flatten(partitions[:i]) + flatten(partitions[i+1:])

        # get the trained supervised model
        priors, posteriers = train_supervised(training_data)

        # accumulate accuracy of test data
        sums += evaluate_supervised(priors, posteriers, test_data)
s
    return sums/k

# This function creates a postierer count dictionary
def construct_posteriers_unsupervised(priors, distributions, training_data):
    # new posteriers
    posteriers = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))

    # convert fractional counts to probabilities
    for instance, dist_dict in zip(training_data, distributions):
        for class_name, dist in dist_dict.items():
            for attribute, value in enumerate(instance):
                posteriers[class_name][attribute][value] += (dist / 
                                                            priors[class_name])

    return posteriers

# This function cleans training data
def clean(training_data):
    cleansed_data = []

    # transpose training data for imputation
    # make columns rows instead for easier access and manipulation
    for column in zip(*training_data):

        # count elements in each column
        # remove '?' character from counts
        # get most common item in column
        counts = Counter(column)
        counts.pop('?', None)
        common = counts.most_common(1)[0][0]

        # imputate invalid data with max occuring data in column
        new_column = [common if val == '?' else val for val in column]
        cleansed_data.append(new_column)

    # transpose modified data back into rows
    return list(map(list, zip(*cleansed_data)))

# This function should build an unsupervised NB model
def train_unsupervised(training_data):

    # get sorted list of classes in data set
    # allowed columns of classes to always be same order
    classes = sorted(set(map(itemgetter(-1), training_data)))

    # strip classes from training data
    # also clean training data
    classless_training = [instance[:-1] for instance in training_data]

    # random_distributions
    distributions = []

    # priors of distribution counts
    priors = defaultdict(float)

    # calculate priors of random distribtions
    # and transform training data into randomized labelled data
    for instance in classless_training:
        rand_distribution = np.random.dirichlet(np.ones(len(classes)), size=1)
        
        # add and create tupled pairs of (class, distribution)
        # add as a ordered dictionary to keep ordering of classes
        class_distribution = list(zip(classes, *rand_distribution))
        distributions.append(OrderedDict(class_distribution))

        # fraction prior counts
        for class_name, dist in class_distribution:
            priors[class_name] += dist

    # transform into posterier probability dictionary
    posteriers = construct_posteriers_unsupervised(priors, 
                                                   distributions, 
                                                   classless_training)

    # return a tuple of all the needed data structures
    return priors, posteriers, distributions, classless_training

# This function builds new predictions from an NB unsupervised model
def predict_unsupervised(priors, posteriers, distributions, training_data):

    # convert fractions to probabilities
    probs = {cs: cnt / len(training_data) for cs, cnt in priors.items()}

    # go over each instance and distribution sumultaneously
    # collect new distributions
    for instance, distribution in zip(training_data, distributions):
        new_dists = []

        # go over each class in distribution
        for class_name in distribution:
            product = 1

            # calculate new distributions
            for attribute, value in enumerate(instance):
                product *= posteriers[class_name][attribute][value]

            # multiply by probability and add it
            product *= probs[class_name]
            new_dists.append(product)

        # normalise distribution into probabilities
        for class_name, new_dist in zip(distribution, new_dists):
            distribution[class_name] = new_dist / sum(new_dists)

    # recreate posteriers with new distributions
    new_posteriers = construct_posteriers_unsupervised(priors, 
                                                       distributions, 
                                                       training_data)

    return new_posteriers, distributions
    
# This function should evaluate a set of predictions, in an unsupervised manner
def evaluate_unsupervised(distributions, training_data):

    # keep a counter of correct instances found
    correct = 0

    # go over each instance ad parallel distribution in training data
    # get predicted class for each instance
    for instance, distribution in zip(training_data, distributions):
        predict_class, _ = max(distribution.items(), key=itemgetter(1))

        # if class is identical to the instances last column
        # increment the count
        if predict_class == instance[-1]:
            correct += 1

    return correct/len(training_data)

# This function gets all the csv files in the current directory
def get_datasets(extension='.csv'):

    files = []

    # go through all items in current directorys
    for file in os.listdir('.'):

        # add if item is a file and ends with extension
        if os.path.isfile(file) and file.endswith(extension):
            files.append(file)

    # return items in sorted order
    return sorted(files)

# This function is the main driver of program
def main():
    datasets = get_datasets()

    for file in datasets:
        data = preprocess(file)

        #imputate invalid data
        data = clean(data)

        # SUPERVISED
        priors, posteriers = train_supervised(data)

        evaluate = evaluate_supervised(priors, posteriers, data)

        print(file)
        print('supervised: %f' % (evaluate))
        print('cross_validation: %f' % (cross_validation(data, 10)))

        # UNSUPERVISED
        priors, posteriers, distributions, training_data = train_unsupervised(data)

        ITERATIONS = 2
        count = 0
        while count < ITERATIONS:
            posterier_temp, distribution_temp = predict_unsupervised(priors,
                                                                     posteriers, 
                                                                     distributions, 
                                                                     training_data)

            posteriers, distributions = posterier_temp, distribution_temp

            count += 1

        evaluate = evaluate_unsupervised(distributions, data)

        print('unsupervised: %f\n' % (evaluate))


if __name__ == '__main__':
    main()
