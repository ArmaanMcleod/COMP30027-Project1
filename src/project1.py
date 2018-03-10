import numpy as np  
import pandas as pd
import csv
from collections import defaultdict
# This function should open a data file in csv, and transform it into a usable format 
def preprocess():
    filepath = "../datasets/car-dos.csv"
  

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
    
    print(probList)
    for i in range(0,len(probList)):
        probList[i] = probList[i].to_dict()
    print(probList)
  
    return probList

# This function should predict the class for a set of instances, based on a trained model 
def predict_supervised(probList):
    
    
    testrow = ['low','low','2','4','big','med','good']
    prob = 1
    for i in range(0,len(testrow)) :
        
        if ('acc',testrow[i]) in probList[i]: 
            prob = prob * probList[i][('acc',testrow[i])]
        else : 
            prob=prob * 0.00001
  
    print(prob)
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

df  = preprocess()
probs = train_supervised()
predict_supervised(probs)

