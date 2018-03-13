import numpy as np  
import pandas as pd
import csv
from pprint import pprint
from collections import defaultdict
K  = 10 # k is the number of pieces to divide the data into

# This function should open a data file in csv, and transform it into a usable format 
def preprocess():
    filepath = "../datasets/hypothyroid-dos.csv"
    

    df = pd.read_csv(filepath, header = None)
    print(df[0].value_counts())
    clean(df)
    print(df[0].value_counts())
  
    
    
    #df.replace('?', np.NaN)
    return df

# This function should build a supervised NB model
def train_supervised(df):
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
def predict_supervised(probList,testrow,classes):
    
    
    
  
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
def evaluate_supervised(testcsv, probs,classes):
    correct = 0
    total = 0 
    #iterate over each of the rows and pass it to the predict supervised method
    for index, testrow in testcsv.iterrows() :
        
        if predict_supervised(probs, testrow.tolist(),classes) == testrow[len(testcsv.columns)-1]:
            correct +=1
        
            
        total +=1
       
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

def k_fold(fulldf):
    #split array into 10 pieces
    karrays = np.array_split(fulldf,K)
    counter = 0 
    sum = 0 #record results
    
    for i in range(0,len(karrays)):
        counter = 0
        testdf = karrays[i] #set the test array as one of the chunks
        
       
        for j in range(0,len(karrays)):
            #ensure that were not adding the test chunk to the array 
            if i != j: 
                
               if counter == 0:
                   counter+=1
                   traindf = karrays[j] #initialise the data to be trained
                   continue 
               #concatinate all the chunks that arent the test chunk
               traindf = pd.concat([traindf,karrays[j]],axis = 0) 
        
        #train the classifer by building the probability dictionary 
        probs, classes = train_supervised(traindf)
        #evaluate the classifier 
    
        sum+= evaluate_supervised(testdf, probs, classes)
    return sum/K       
               
           
            
#cleaning methiod removes '?' and places in it the most common value for that column
def clean(dataframe):
    for index, testrow in dataframe.iterrows():
      for i in range(0,len(testrow)):
          if testrow[i] == '?':
              testrow[i] = dataframe[i].value_counts().idxmax()
  
    return
     

def driver():
    fulldf = preprocess()
    
    print(k_fold(fulldf))
    #train the data
    
def non_kfold_driver():
    filepath = "../datasets/hypothyroid-dos.csv"

    df = pd.read_csv(filepath, header = None)
    #df.replace('?', np.NaN)
    probs, classes = train_supervised(df)
    
    print(evaluate_supervised(df, probs,classes))

driver()
non_kfold_driver()



