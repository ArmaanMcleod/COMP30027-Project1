import numpy as np  
import pandas as pd
import csv
from pprint import pprint
from collections import defaultdict

# This function should open a data file in csv, and transform it into a usable format 
def preprocess():
    filepath = "../datasets/car-dos.csv"
  

    df = pd.read_csv(filepath, header = None)
    #df.replace('?', np.NaN)
    return df

# This function should build a supervised NB model
def train_supervised():
    #prior probabilties
    highProb = df.groupby(len(df.columns)-1).size().div(len(df))
     
    #probabiltiy deictionary 
    probList = defaultdict(lambda: defaultdict(float))
    
    #this loop calculates the probabilties 
    for i in range(0,len(df.columns)-1):
        cols =[len(df.columns)-1,i]
        trained = df.groupby(cols).size().div(len(df)).div(highProb, axis = 0,level=(len(df.columns)-1))
        probList[i] = trained
    
    #converts dataframe into dictionary 
    for i in range(0,len(probList)):
        probList[i] = probList[i].to_dict()
        #collects the classes into an list
    classes = df[len(df.columns)-1].unique()
    return probList, classes

# This function should predict the class for a set of instances, based on a trained model 
def predict_supervised(probList,testrow):
    
    
    
  
    classChance = list() #keep record of all classchanes for this row
    for possibleClass in classes : 
        prob =1  #begin at 1 
        for i in range(0,len(testrow)) : #for every element multiply in 
        
            if (possibleClass,testrow[i]) in probList[i]: 
                prob = prob * probList[i][(possibleClass,testrow[i])]
            else : 
                prob=prob * 0.000000000001 #epsilon 
        classChance.append(prob) #add the item to the list
  
            #here we get the highest probability and match it to the class
    return classes[classChance.index(max(classChance))]

# This function should evaluate a set of predictions, in a supervised context 
def evaluate_supervised(testcsv):
    correct = 0
    total = 0 
    #iterate over each of the rows and pass it to the predict supervised method
    for index, testrow in testcsv.iterrows() :
        print(testrow.tolist())
        if predict_supervised(probs, testrow.tolist()) == testrow[len(testcsv.columns)-1]:
            correct +=1
        
            
        total +=1
        print(correct/total)
    return correct/total

# This function should build an unsupervised NB model 
def train_unsupervised():
    return

# This function should predict the class distribution for a set of instances, based on a trained model
def predict_unsupervised():
    return

# This function should evaluate a set of predictions, in an unsupervised manner
def evaluate_unsupervised():
    return

#load in test data
testdata = "../datasets/car-dos.csv"
testcsv = pd.read_csv(testdata, header = None)
#preprocess the data
df = preprocess()
#train the data
probs, classes = train_supervised()
#evaluate the data 

evaluate_supervised(testcsv)


