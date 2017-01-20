# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 08:35:29 2016

@author: mounik
"""
"""Relevant Libraries"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

def PlotConfusionMatrix(conf_matrix, target_labels, title="Confusion Matrix", cmap = plt.cm.Greens):
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

def PerformLogisticRegression(X_train,y_train, X_test, y_test):
    # This function performs Logistic Regression and displays the results
    logReg = linear_model.LogisticRegression(penalty='l1', max_iter = 1000, solver='liblinear')
    logReg.fit(X_train, y_train)
    y_pred = logReg.predict(X_test)
    print("Logistic regression\n%s\n" % (
    metrics.classification_report(
        y_test,
        y_pred)))
    print("Accuracy of algorithm: "+str(metrics.accuracy_score(y_test,y_pred)*100))
    conf_matrix = []
    target_labels = y_test.unique()
    conf_matrix = metrics.confusion_matrix(y_test, y_pred, labels =target_labels)
    print(conf_matrix)
    # Used to plot confusion matrix for better visualisation
    PlotConfusionMatrix(conf_matrix, target_labels)
    
    

def AddFeatures(X):
    # This function adds features to the dataset so that a good decision boundary can be obtained for generalisation
    X['X12'] = X['SAS']**2
    X['X22'] = X['Open']**2
    X['X32'] = X['High']**2
    X['X42'] = X['Low']**2
    X['X52'] = X['Close']**2
    X['X62'] = X['Volume']**2
    X['X1X2'] = X['SAS'] * X['Open']
    X['X1X3'] = X['SAS'] * X['High']
    X['X1X4'] = X['SAS'] * X['Low']
    X['X1X5'] = X['SAS'] * X['Close']
    X['X1X6'] = X['SAS'] * X['Volume']
    X['X2X3'] = X['Open'] * X['High']
    X['X2X4'] = X['Open'] * X['Low']
    X['X2X5'] = X['Open'] * X['Close']
    X['X2X6'] = X['Open'] * X['Volume']
    X['X3X4'] = X['High'] * X['Low']
    X['X3X5'] = X['High'] * X['Close']
    X['X3X6'] = X['High'] * X['Volume']
    X['X4X5'] = X['Low'] * X['Close']
    X['X4X6'] = X['Low'] * X['Volume']
    X['X5X6'] = X['Close'] * X['Volume']
    return X

def NormaliseAndSplitDataFrame(df_merge_sas_price):
    # This function normalises the required columns in a data frame and splits them to parameteres and the target variable
    symbolList = []
    symbolList = df_merge_sas_price['symbol'].unique()
    colsToNormalise = df_merge_sas_price.columns.tolist()
    colsToNormalise.pop(0)
    colsToNormalise.pop(1)
    for symbol in symbolList:
        colsToNormalise.remove(symbol)
    colsToNormalise.remove('HiLoPr')
    df_merge_sas_price[colsToNormalise] = df_merge_sas_price[colsToNormalise].apply(lambda x:(x-x.mean())/(x.max()-x.min()))
    X = df_merge_sas_price[colsToNormalise]
    X = AddFeatures(X)
    X = pd.concat([X, df_merge_sas_price[symbolList]], axis = 1)
    y = df_merge_sas_price['HiLoPr']
    return X,y
        
def MergeDataFrames(df_sas_lt, df_price_data):
    df_sas_lt = df_sas_lt[(df_sas_lt['date'].dt.hour == 9) & (df_sas_lt['date'].dt.minute == 30)]
    df_price_data = df_price_data[(df_price_data['date'].dt.hour == 9) & (df_price_data['date'].dt.minute == 30)] 
    df_merge_sas_price= pd.merge(df_sas_lt,df_price_data, on=['symbol','date'])
    print(df_merge_sas_price.head())
    # The price is said to be high(1) if the closing price of the previous day is lesser than the closing price of present day
    # and price is said to be low(0) if the closing price of the previous days is more than the closing price of present day
    lstPrice = []
    for ind in range(len(df_merge_sas_price['Close'])):
        if ind==0:
            lstPrice.append(0) # The first entry be default is 0 since there is no closing price for the previous day
        elif((df_merge_sas_price['Close'][ind] > df_merge_sas_price['Close'][ind-1]) and (df_merge_sas_price['symbol'][ind] == df_merge_sas_price['symbol'][ind-1])):
            lstPrice.append(1)
        elif((df_merge_sas_price['Close'][ind] <= df_merge_sas_price['Close'][ind-1]) and (df_merge_sas_price['symbol'][ind] == df_merge_sas_price['symbol'][ind-1])):
            lstPrice.append(0)
        elif(df_merge_sas_price['symbol'][ind] != df_merge_sas_price['symbol'][ind-1]):
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
    #df_merge_sas_price.to_csv('AllSASAndPrice.csv', sep=',', encoding = 'utf-8')
    df_symbol = pd.DataFrame()
    df_symbol = pd.get_dummies(df_merge_sas_price['symbol'])
    df_merge_sas_price = pd.concat([df_merge_sas_price, df_symbol], axis = 1)
    return df_merge_sas_price   
    
    
def LoadDataFrames(pathSAS_data,pathPrice_Data):
    # This function loads the data frames from the paths passed as parameters
    df_sas_lt = pd.DataFrame()
    dfList = []
    lstCompNames = []
    for files in glob.glob(pathSAS_data):
        df = pd.read_csv(files, sep =',')
        dfList.append(df)
        compName =files[:-4]
        compName = compName[18:]
        lstCompNames.append(compName)
        
    df_sas_lt =pd.concat(dfList,axis =0)
    #del df_sas_lt['date']
    #del df_sas_lt['symbol']
    #df_sas_lt.rename(columns={'date_0':'date'}, inplace = True)
    df_sas_lt['date']= pd.to_datetime(df_sas_lt['date'])
    #df_sas_lt.to_csv('Lt_data.csv', sep =',', encoding='utf-8')
    
    dfPriceList = []
    for files in glob.glob(pathPrice_Data):
        df = pd.read_csv(files, sep=',')
        compPriceName = files[29:]
        compPriceName = compPriceName[:-10]        
        df['symbol'] = compPriceName
        dfPriceList.append(df)
    df_price_data = pd.concat(dfPriceList, axis = 0)
    df_price_data.rename(columns={'Unnamed: 0': 'date'}, inplace = True)
    df_price_data['date']= pd.to_datetime(df_price_data['date'])
    print(df_sas_lt.head())
    print(df_price_data.head())
    return df_sas_lt, df_price_data

if __name__=='__main__':
    
    # Location of the SAS_data file in the interns1 folder
    pathSAS_data = 'interns1/SAS_data/*_lt.csv'
    
    # Location of the price_data file in the interns1 folder
    pathPrice_Data = 'new_data/interns1/price_data/*_ohlcv.csv'
    
    # Load the data frames from the above locations
    df_sas_lt, df_price_data = LoadDataFrames(pathSAS_data, pathPrice_Data)
    
    # Merges  Dataframe to our desired condition
    df_merge_sas_price = MergeDataFrames(df_sas_lt, df_price_data)
#    
    # Normalise data and split the data frames to X-parameters and y-target
    X,y = NormaliseAndSplitDataFrame(df_merge_sas_price)
##    
    X_train,X_test,y_train, y_test = train_test_split(X, y,test_size= 0.25)
##    #X_train,X_cross_val,y_train, y_cross_val = train_test_split(X_train, y_train, test_size = 0.15)
##    
#    # Performs logistic regression and displays accuracy, precision and f1score of each class
    PerformLogisticRegression(X_train, y_train, X_test, y_test)
    