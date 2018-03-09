import numpy as np  
import pandas as pd
import csv
# This function should open a data file in csv, and transform it into a usable format 
def preprocess():
    filepath = "../datasets/car-dos.csv"
  

    df = pd.read_csv(filepath, header = None)
    return df

# This function should build a supervised NB model
def train_supervised():
    print(df.groupby(len(df.columns)-1).size().div(len(df)))
    return

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

preprocess()

