# -*- coding: utf-8 -*-
"""
Created on Fri May 13 00:58:06 2016

@author: Mounik Muralidhara
"""

import sys
import math
import numpy
import csv
from yahoo_finance import yahoo_finance
import pprint

def GenerateGraphEdges(lstDistMatrix, lstValidCompanies):
    # The function generates all the edges possible from the distance matrix
    lstEdgeDetails = []
    for singRow in lstDistMatrix:
        #for index in range(1,len(singRow)):
        for ind in range(1, len(singRow)):
            if(singRow[ind]>0):
                lstTempEdgeDetails = []
                lstFirstPartEdge = []
                lstFirstPartEdge.append(singRow[0])
                lstFirstPartEdge.append(lstValidCompanies[ind])
                lstTempEdgeDetails.append(lstFirstPartEdge)
                lstTempEdgeDetails.append(singRow[ind])
                lstEdgeDetails.append(lstTempEdgeDetails)
    return sorted(lstEdgeDetails, key= lambda x: x[1])
            

def GenerateDistanceMatrix(lstCoVarMatrix):
    # The function converts the covariance matrix to a distance matrix
    for aList in lstCoVarMatrix:
        for index in range(1, len(aList)):
            aList[index] = 1-abs(aList[index])
    return lstCoVarMatrix

def GenerateCovarianceMatrix(lstCoVarMatrixResult, lstValidCompanies):
    # This function generates the covarince matrix for all the companies
    lstCoVarMatrix = []
    for index in range(len(lstValidCompanies)):
        tempRowList = []
        tempRowList.append(lstValidCompanies[index])
        for ind in range(len(lstValidCompanies)):
            for item in lstCoVarMatrixResult:
                if(lstValidCompanies[index] == item[0] and lstValidCompanies[ind] == item[1]):
                    tempRowList.append(item[2])
        lstCoVarMatrix.append(tempRowList)                        
    return lstCoVarMatrix
    
def DetermineSpearmanRho(sumOfDifferences,totalNumberOfElements):
    # This function determines spearman rho for sum of differences and total number of elements
    rho = 0
    rho = 1-((6*sumOfDifferences)/((totalNumberOfElements)*(math.pow(totalNumberOfElements,2)-1)))
    return rho

def CalculateSpearmanRhoCoVar(lstAllCompCloseDataRank):
    # This function calaculates the Spearman Rho Covariance and gives out possible edges
    lstCoVarMatrixResult = []
    for index in range(len(lstAllCompCloseDataRank)):
        for ind in range(index,len(lstAllCompCloseDataRank)):
            sumOfDifferences = 0
            tempListCoVar = []
            tempListCoVarInv = []
            tempListCoVar.append(lstAllCompCloseDataRank[index][0])
            tempListCoVar.append(lstAllCompCloseDataRank[ind][0])
            tempListCoVarInv.append(lstAllCompCloseDataRank[ind][0])
            tempListCoVarInv.append(lstAllCompCloseDataRank[index][0])
            lstTempDifferences = []
            lstTempDifferences = numpy.array(lstAllCompCloseDataRank[index][1])-numpy.array(lstAllCompCloseDataRank[ind][1])
            lstTempDifferences = [i**2 for i in lstTempDifferences]
            sumOfDifferences = sum(lstTempDifferences)
            rho = DetermineSpearmanRho(sumOfDifferences,len(lstAllCompCloseDataRank[index][1]))
            tempListCoVar.append(rho)
            tempListCoVarInv.append(rho)
            if(lstAllCompCloseDataRank[index][0] == lstAllCompCloseDataRank[ind][0]):
                lstCoVarMatrixResult.append(tempListCoVar)
            else:
                lstCoVarMatrixResult.append(tempListCoVar)
                lstCoVarMatrixResult.append(tempListCoVarInv)
    return lstCoVarMatrixResult
    
def GenerateRanks(dictAllCompCloseData):
    # This function generates the ranks based on the first differences of the company
    lstAllCompCloseDataRank = []
    for comp in dictAllCompCloseData:
        rankList = []
        rankList =(-numpy.array(dictAllCompCloseData[comp])).argsort().argsort()
        rankList = [x+1 for x in rankList]
        lstTempRankList = []
        lstTempRankList.append(comp)
        lstTempRankList.append(rankList)
        lstAllCompCloseDataRank.append(lstTempRankList)
    return lstAllCompCloseDataRank

def CalculateFirstDifferences(dictAllCompCloseData):
    # This function is used to calculate the first differences of the closing data acquired over a period
    lstValidCompanies = []
    for item in dictAllCompCloseData:
        lstFirstDiff = []
        for index in range(len(dictAllCompCloseData[item])-1):
            lstFirstDiff.append(dictAllCompCloseData[item][index+1]-dictAllCompCloseData[item][index])
        dictAllCompCloseData[item] = lstFirstDiff
        lstValidCompanies.append(item)
    return lstValidCompanies, dictAllCompCloseData        

def GetHistoricalData(companyList):
    # This function gets the historical data of closing prices utilizing yahoo_finance package
    dictAllCompCloseData = {}
    for compName in companyList:
        lstCompCloseDat = []
        compDetails = yahoo_finance.Share(compName)
        all_data  = (compDetails.get_historical('2015-01-01', '2016-05-10'))
        for dictItem in all_data:
            lstCompCloseDat.append(math.log(float(dictItem['Close'])))
        if(len(lstCompCloseDat) != 0):
            dictAllCompCloseData[compName] = lstCompCloseDat
    pprint(dictAllCompCloseData)
    return dictAllCompCloseData
            
            
if __name__ == '__main__':
    inputData = open("Quiz_Ticker_universe.csv").read().split()
    companyList =[]
    for line in inputData:
        companyList.append(line)
    
    dictAllCompCloseData = {}
    # This function gets the historical data of closing prices utilizing yahoo_finance package
    dictAllCompCloseData = GetHistoricalData(companyList)
    
    lstValidCompanies = []
    # This function is used to calculate the first differences of the closing data acquired over a period
    lstValidCompanies,dictAllCompCloseData = CalculateFirstDifferences(dictAllCompCloseData)
    
    lstAllCompCloseDataRank = []
    # This function generates the ranks based on the first differences of the company
    lstAllCompCloseDataRank = GenerateRanks(dictAllCompCloseData)
    
    lstCoVarMatrixResult = []
    # This function calaculates the Spearman Rho Covariance and gives out possible edges
    lstCoVarMatrixResult=CalculateSpearmanRhoCoVar(lstAllCompCloseDataRank)
    
    lstCoVarMatrix = []
    # This function generates the covarince matrix for all the companies
    lstCoVarMatrix = GenerateCovarianceMatrix(lstCoVarMatrixResult, lstValidCompanies)
    
    lstValidCompnyRef = lstValidCompanies
    lstValidCompanies.insert(0,' ')
    
    # Write the correlation to corr.csv file
    with open('corr.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(lstValidCompanies)
        writer.writerows(lstCoVarMatrix)
    
    lstDistMatrix = []
    # The function converts the covariance matrix to a distance matrix
    lstDistMatrix = GenerateDistanceMatrix(lstCoVarMatrix)
    
    # Write the distance matrix to dist.csv file
    with open('dist.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(lstValidCompanies)
        writer.writerows(lstDistMatrix)
    
    lstEdgeDetails = []
    # The function generates all the edges possible from the distance matrix
    lstEdgeDetails =  GenerateGraphEdges(lstDistMatrix, lstValidCompanies)