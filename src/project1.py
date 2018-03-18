import numpy as np  
import pandas as pd
import csv
from pprint import pprint
from collections import defaultdict
from random import shuffle






K  = 10 # k is the number of pieces to divide the data into must be atleast 2

# This function should open a data file in csv, and transform it into a usable format 


def unsuper_driver():
    unsuper_df, classes = unsupervised_preprocess()
    probList, classes, priors = train_unsupervised(unsuper_df, classes)
    testdf = pd.read_csv(filepath, header = None)
   
    return evaluate_unsupervised(testdf,probList,classes,priors)



def preprocess():
    
    
    
    df = pd.read_csv(filepath, header = None)

    clean(df) #impute missing values if they exist
    df = df.sample(frac=1).reset_index(drop=True) #shuffles the dataframe
  
    
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
       
        trained = df.groupby(cols).size().div(len(df)).multiply(highProb, axis = 0,level=(len(df.columns)-1))
        
        probList[i] = trained
 

    #converts dataframe into dictionary 
    for i in range(0,len(probList)):
        probList[i] = probList[i].to_dict()
        #collects the classes into an list
    classes = df[len(df.columns)-1].unique()
    
    priors = get_super_priors(df,classes)
    
    return probList, classes,priors

# This function should predict the class for a set of instances, based on a trained model 
def predict_supervised(probList,testrow,classes,priors):
    
    classChance = [0.0] * len(classes)#keep record of all classchanes for this row
    classlist = classes.tolist()
    for possibleClass in classes : 
        
        prob =1 *priors[classlist.index(possibleClass)]
        for i in range(0,len(testrow)-1) : #for every element multiply in 
  
            if (possibleClass,testrow[i]) in probList[i]: 
               
                prob = prob * probList[i][(possibleClass,testrow[i])]
            else : 
               
                prob = prob * 0.00000000000001 #epsilon 
        classChance[classlist.index(possibleClass)]= prob
        
        

            #here we get the highest probability and match it to the class
    

    return classes[classChance.index(max(classChance))]

# This function should evaluate a set of predictions, in a supervised context 
def evaluate_supervised(testcsv, probs,classes,priors):

    correct = 0
    total = 0 
    #iterate over each of the rows and pass it to the predict supervised method
    for index, testrow in testcsv.iterrows() :
        
        
        if predict_supervised(probs, testrow.tolist(),classes,priors) == testrow[len(testcsv.columns)-1]:
            
            correct +=1
     
        total +=1
        
     
    return correct/total

# This function should build an unsupervised NB model 
def train_unsupervised(df, classes):
   

    priors= get_priors(df,classes)
    
        

    probList = make_probability_dictionary(df,classes)
 
    df = assign_distro(df,classes,probList,priors)
   
    probList = make_probability_dictionary(df,classes)
 
    df = assign_distro(df,classes,probList,priors)
   
   

    return probList, classes, priors
    
def make_probability_dictionary(df,classes):
     probList = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.001)))
     for prob_class in range(0,len(classes)):
        for index, row in df.iterrows() :
            for i in range(0,len(row)-len(classes)):
                probList[i][classes[prob_class]][row[i]] +=row[len(row)-len(classes)+prob_class]
     return probList
    
def assign_distro(df,classes,probList,classProbs):
     for index, row in df.iterrows():
       
       
       for probable_Class in range(len(classes)):
        
           product = 1
           
           for attrib in range((len(row)-len(classes))):    
              
               product = product * classProbs[probable_Class] *probList[attrib][classes[probable_Class]][row[attrib]]
               
           df.loc[index,df.columns[(len(row)-len(classes) + probable_Class)]] = product
    
     normalise_unsupervised(df,classes)
     
     return df
    
def get_priors(df,classes):
    classProbs = []
    for i in classes:   
        classProbs.append(sum(df[i])/len(df))
    
    return classProbs



# This function should predict the class distribution for a set of instances, based on a trained model
def predict_unsupervised(probList,testrow,classes,priors):
        
    shuffle(classes)
    classChance = [0.0] * len(classes)#keep record of all classchanes for this row
    classlist = classes.tolist()
    for possibleClass in classes : 
        
        prob =1 *priors[classlist.index(possibleClass)]
        for i in range(0,len(testrow)-len(classes)) : #for every element multiply in 
  
            if testrow[i] in probList[i][possibleClass]: 
              
                prob = prob * probList[i][possibleClass][testrow[i]]
            else: 
               
               prob = prob * 0.0000000001 #epsilon 
        classChance[classlist.index(possibleClass)]= prob
        
        

    
    return classes[classChance.index(max(classChance))]
  

# This function should evaluate a set of predictions, in an unsupervised manner
def evaluate_unsupervised(testcsv, probs,classes,priors):
    
    correct = 0
    total = 0 
    #iterate over each of the rows and pass it to the predict supervised method
    for index, testrow in testcsv.iterrows() :
        
            
        
        if predict_unsupervised(probs, testrow.tolist(),classes,priors) == testrow[len(testcsv.columns)-1]:
            
            correct +=1
     
        total +=1
        
     
    return correct/total
   

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
        probs, classes, priors = train_supervised(traindf)
        #evaluate the classifier 
    
        sum+= evaluate_supervised(testdf, probs, classes,priors)
        
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

    df = pd.read_csv(filepath, header = None)
  
    probs, classes,priors = train_supervised(df)
    
    print(evaluate_supervised(df, probs,classes,priors))

def unsupervised_preprocess():
      df = pd.read_csv(filepath, header = None)
      
      clean(df) #impute missing values if they exist
      df = df.sample(frac=1).reset_index(drop=True) #shuffles the dataframe
      classes = df[len(df.columns)-1].unique()
      
      df = df.iloc[:, :-1]
      for probable_class in classes:
          df[probable_class] = pd.Series(np.random.rand(len(df)))
    
      normalise_unsupervised(df,classes)
   
      return df, classes
        
def normalise_unsupervised(df, classes):
    for index, row in df.iterrows():
        for i in range((len(row)-len(classes)),len(row)):
            temp= float(row[i])/sum(row[-len(classes):])
            df.loc[index,df.columns[i] ] = temp
      
   
    return

def get_super_priors(df,classes):
    priors = [0.0] * len(classes)
    total = 0
    classes = classes.tolist()
    for index, row in df.iterrows():
        priors[classes.index(row[len(row)-1])] += 1 
        total += 1
    for i in range(len(priors)):
        priors[i] = priors[i]/total

           
      
   
    return priors
    
    
filepath = "../datasets/hypothyroid-dos.csv"
non_kfold_driver()
driver()

print(unsuper_driver())



