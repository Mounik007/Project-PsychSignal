# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:26:10 2016

@author: mounik
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from sklearn.tree import DecisionTreeClassifier, export_graphviz
        
def PlotConfusionMatrix(conf_matrix, target_labels, title="Confusion Matrix", cmap = plt.cm.Blues):
    # Used to plot confusion matrix for visualisation purposes
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_labels))
    plt.xticks(tick_marks, target_labels, rotation=45)
    plt.yticks(tick_marks, target_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def VisualizeDecisionTree(tree, colName, target_labels):
    # Used to write and draw the graphs
    tgt_Labels = []
    for label in target_labels:
        tgt_Labels.append(str(label))
    with open("dt.dot", "w") as f:
        export_graphviz(tree, out_file=f, feature_names = colName, class_names = tgt_Labels)
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run graphviz ")

def PerformDecisionTreeClassification(X_train, X_test, y_train, y_test, colName):
    # This function performs Decision Tree Classification and displays the results
    # It also writes and draws the decision tree
    target_labels = y_test.unique()
    dt = DecisionTreeClassifier(criterion="entropy", min_samples_split = 20, random_state = 99)
    dt.fit(X_train, y_train)
    VisualizeDecisionTree(dt, colName, target_labels)
    y_pred = dt.predict(X_test)
    print("Decision Tree\n%s\n" % (
    metrics.classification_report(
        y_test,
        y_pred)))
    print("Accuracy of algorithm: "+str(metrics.accuracy_score(y_test,y_pred)*100))
    conf_matrix = []
    conf_matrix = metrics.confusion_matrix(y_test, y_pred, labels =target_labels)
    print(conf_matrix)
    # Used to plot confusion matrix for better visualisation
    PlotConfusionMatrix(conf_matrix, target_labels)    
                    
def RoundOffNumericalValuesAttributes(df_merge_sas_price):
    numericalCols = ['SAS', 'Open', 'High', 'Low', 'Close','Volume']
    #decimals = pd.Series([-1,-1,-1,-1,-1], index = numericalCols)
    for colName in numericalCols:
        lstTempSet = []
        lstTempSet = df_merge_sas_price[colName]
        if(colName == 'Volume'):
            lstTempSet = [round(elem,-5) for elem in lstTempSet]
        elif(colName != 'SAS'):
            lstTempSet = [round(elem,-1) for elem in lstTempSet]
        else:
            lstTempSet = [round(elem,1) for elem in lstTempSet]
        df_merge_sas_price[colName] = lstTempSet
    return df_merge_sas_price, numericalCols
        
def MergeDataFrames(df_sas_lt, df_price_data):
    df_sas_lt = df_sas_lt[(df_sas_lt['date'].dt.hour == 9) & (df_sas_lt['date'].dt.minute == 30)]
    df_price_data = df_price_data[(df_price_data['date'].dt.hour == 9) & (df_price_data['date'].dt.minute == 30)] 
    df_merge_sas_price= pd.merge(df_sas_lt,df_price_data, on='date')
    #df_merge_sas_price = df_price_data
    # The price is said to be high(1) if the closing price of the previous day is lesser than the closing price of present day
    # and price is said to be low(0) if the closing price of the previous days is more than the closing price of present day
    lstPrice = []
    for ind in range(len(df_merge_sas_price['Close'])):
        if ind==0:
            lstPrice.append(0) # The first entry be default is 0 since there is no closing price for the previous day
        elif(df_merge_sas_price['Close'][ind] > df_merge_sas_price['Close'][ind-1]):
            lstPrice.append(1)
        else:
            lstPrice.append(0)
    
    # The below calcualtion is necessary as it helps us to evaluate of how well our machine learnining model performs
    counter = 0.0
    for ind in range(len(lstPrice)):
        if(lstPrice[ind] == 1):
            counter+=1    
    total_1_percent = (counter/len(lstPrice))*100
    total_0_percent = 100-total_1_percent
    print("Percentage of entries falling under class High(1) Price: "+str(total_1_percent))
    print("Percentage of class falling under class Low(0) Price: "+str(total_0_percent))
    df_merge_sas_price['HiLoPr'] = lstPrice
    # Saves the current data frame to a csv file
    df_merge_sas_price.to_csv('SASAndPrice.csv', sep=',', encoding = 'utf-8')
    return df_merge_sas_price   
    
    
def LoadDataFrames(pathSAS_data,pathPrice_Data):
    # This function loads the data frames from the paths passed as parameters
    df_sas_lt = pd.read_csv(pathSAS_data, sep=',')
    df_sas_lt['date']= pd.to_datetime(df_sas_lt['date'])
    df_price_data = pd.read_csv(pathPrice_Data, sep=',')
    df_price_data['date']= pd.to_datetime(df_price_data['date'])
    return df_sas_lt, df_price_data

if __name__=='__main__':
    
    # Location of the SAS_data file in the interns1 folder
    pathSAS_data = 'interns1/SAS_data/SPY_lt.csv'
    
    # Location of the price_data file in the interns1 folder
    pathPrice_Data = 'interns1/price_data/SPY_ohlcv.csv'
    
    # Load the data frames from the above locations
    df_sas_lt, df_price_data = LoadDataFrames(pathSAS_data, pathPrice_Data)
    
    # Merges  Dataframe to our desired condition
    df_merge_sas_price = MergeDataFrames(df_sas_lt, df_price_data)
    
    # Rounding the numbers helps us get better accuracy
    numericalCols = []
    df_merge_sas_price, numericalCols = RoundOffNumericalValuesAttributes(df_merge_sas_price)
    
    # Splitting the database into training set and testing dataset
    colName = ['SAS', 'Open', 'High', 'Low', 'Close','Volume']
    X = df_merge_sas_price[colName]
    y = df_merge_sas_price['HiLoPr']
    X_train,X_test,y_train, y_test = train_test_split(X, y,test_size= 0.15)
    
    #Performs decision tree classification algorithm, writes, draws decision tree and displays the result
    PerformDecisionTreeClassification(X_train, X_test, y_train, y_test, colName)