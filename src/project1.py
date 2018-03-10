import numpy as np  
import pandas as pd
import csv
from collections import defaultdict
# This function should open a data file in csv, and transform it into a usable format 
def preprocess():
    filepath = "../datasets/cars-train.csv"


    df = pd.read_csv(filepath, header = None)
    return df

# This function should build a supervised NB model
def train_supervised():
    highProb = df.groupby(len(df.columns)-1).size().div(len(df))
     
    print(highProb)
    probList = defaultdict(lambda: defaultdict(float))
    
    
    for i in range(0,len(df.columns)-1):
        cols =[len(df.columns)-1,i]
        trained = df.groupby(cols).size().div(len(df)).div(highProb, axis = 0,level=(len(df.columns)-1))
        probList[i] = trained
    
  
    for i in range(0,len(probList)):
        probList[i] = probList[i].to_dict()

    classes = df[len(df.columns)-1].unique()
    return probList, classes

# This function should predict the class for a set of instances, based on a trained model 
def predict_supervised(probList,testrow):
    
    

  
    classChance = list()
    for possibleClass in classes : 
        prob =1 
        for i in range(0,len(testrow)) :
        
            if (possibleClass,testrow[i]) in probList[i]: 
                prob = prob * probList[i][(possibleClass,testrow[i])]
            else : 
                prob=prob * 0.00001
        classChance.append(prob)
  
  
    return classes[classChance.index(max(classChance))]

# This function should evaluate a set of predictions, in a supervised context 
def evaluate_supervised(testcsv):
    correct = 0
    total = 0 
    
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

testdata = "../datasets/cars-test.csv"
testcsv = pd.read_csv(testdata, header = None)

df = preprocess()
probs, classes = train_supervised()
evaluate_supervised(testcsv)

