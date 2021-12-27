import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from random import randint
from datetime import datetime
from enum import Enum
import os
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

class LineaReggressionMetrics(Enum):
    R_SQUARED = 1
    ROOT_MEAN_SQUARED_ERROR = 2

def load_dataset(dataset='./Dataset/ico_data_final.csv'):
    all_data = np.genfromtxt(fname=dataset, names=True, delimiter=',', encoding="utf8")
    featureNames = all_data.dtype.names[1:-1]
    ico_names = np.genfromtxt(fname=dataset, delimiter=',', usecols=0, dtype=str,skip_header=1, encoding="utf8")
    all_data = np.genfromtxt(fname=dataset, delimiter=',',skip_header=1, encoding="utf8")[:,1:]
    x = np.genfromtxt(fname=dataset, delimiter=',',skip_header=1, encoding="utf8")[:,1:-1]
    y = np.genfromtxt(fname=dataset, delimiter=',',skip_header=1, encoding="utf8")[:,-1]
    return (featureNames,x,y)




def createFolderIfDoesntExist(folderName):
    exists = os.path.isdir(folderName)
    if not exists:
        os.makedirs(folderName)
        
    return exists

def createResultsFolderIfDoesntExist(folder):
    folderExists = createFolderIfDoesntExist(folder)
    
def makePrediction(model,example_to_predict):
    encoded_x = encodeSingleElement(x,example_to_predict)
    y_pred = model.predict(encoded_x.reshape(1, -1))
    
    return y_pred

def getCovarianceMatrixAndPrintScatterPlot(x,y,saveToFile=False):
    nrows = x.shape[0]
    ncols = x.shape[1]

    for i in np.arange(ncols):
        corCoef = np.corrcoef(x[:,i], y) 
        plt.xlabel(featureNames[i])
        plt.ylabel("Price after 6 months(in $) ")
        plt.suptitle('Scatter Plot of feature {:s} vs Price after 6 months'.format(featureNames[i]))
        corr = "Correlation Coefficient: "+str(corCoef[0,1])
        plt.title(corr)
        plt.scatter(x[:,i], y)
        if(saveToFile == True):
            createFolderIfDoesntExist("images/")
            plt.savefig('images/{:s}_vs_Price_scatter_plot.png'.format(featureNames[i]))
        else: 
            plt.show()
        
        plt.clf() 
    
def plotExpectedVsPredictedOutput(y_test,y_pred,fileName='',saveToFile=False):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(y_test, y_pred,alpha=0.8,edgecolors=(0, 0, 0),s=30)
    ax.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'k--', lw=2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(fileName)
    if(saveToFile == True):
        createFolderIfDoesntExist("images/")
        plt.savefig('images/'+fileName+".png")
    else: 
        plt.show()


createResultsFolderIfDoesntExist("results/")

featureNames,x,y = load_dataset()

getCovarianceMatrixAndPrintScatterPlot(x,y)

from sklearn.preprocessing import OneHotEncoder

def encodeData(x):
    enc = OneHotEncoder(categorical_features='all',
           handle_unknown='error', sparse=False)


    encodedCategoryArray= enc.fit_transform(x[:,8:12])
    allInputsExceptCategorical = np.delete(x, np.s_[8:12], axis=1)
    encodedX = np.concatenate((allInputsExceptCategorical,encodedCategoryArray),axis=1)
    return encodedX

def encodeSingleElement(x,sample):
    enc = OneHotEncoder(categorical_features='all',
           handle_unknown='error', sparse=False)


    encodedCategoryArray= enc.fit(x[:,8:12])
    allInputsExceptCategorical = np.delete(sample, np.s_[8:12], axis=0)
    
    encodedCategories  = enc.transform(sample[8:12].reshape(1, -1))
    encodedX = np.concatenate((allInputsExceptCategorical,encodedCategories.flatten()),axis=0)
    return encodedX