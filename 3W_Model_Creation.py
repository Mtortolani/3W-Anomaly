#import relevant modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import tslearn
from tslearn.metrics import dtw
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeries 
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from random import sample
import numpy as np
from scipy import stats




#extract only sample of full data for testing, scrap data outside of sample
def sampler(total_data, percent_kept):
    X, X_scrap, y, y_scrap = train_test_split(total_data[0], total_data[1], train_size=percent_kept, random_state=42)
    return (X, y)





def knn_f1_score(sample_size=0.001, runs=10):
    '''
    Collects a list of f1 scores for knn clustering for different values of k (always odd)
    Input: int, int
    Output: list
    '''
    total_data_sample = sampler(total_data, sample_size)

    print(f"The length of the sample dataset:", len(total_data_sample[0]))
    
    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(total_data_sample[0], total_data_sample[1], train_size=0.8,random_state=42)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    #learning and reporting of f1 scores
    f1_list = []
    for k in range(0, runs):
        model = KNeighborsTimeSeriesClassifier(n_neighbors = (1+2*k), metric="dtw")
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        expected = y_test
        score = f1_score(expected, predicted, average="weighted")
        f1_list.append(score)
        print(f"Model {1+2*k} ready!")
    return f1_list

    
    
    
#graph out the effect of k on the model at (sample_size=0.001)
scores = knn_f1_score(sample_size=0.001, runs=25)
x = [1 + 2*i for i in range(25)]

#plot in high definition, save file
sns.set_theme(style="whitegrid")
sns.barplot(x, scores, color="b")
plt.rcParams['figure.dpi'] = 3000
plt.rcParams['savefig.dpi'] = 3000
plt.xlabel("K Neighbors Considered")
plt.ylabel("F1 Score")
sns.despine()




def knn_report(sample_size=0.001,k=1):
    #keep a portion of the data for the analysis, ignore the rest
    total_data_sample = sampler(total_data, sample_size)
    print(f"The length of the sample dataset:", len(total_data_sample[0]))

    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(total_data_sample[0], total_data_sample[1], train_size=0.8,random_state=42)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    #knn learning
    model = KNeighborsTimeSeriesClassifier(n_neighbors = k, metric="dtw")
    model.fit(X_train, y_train)
    print("Machine Learning Model ready!")
    predicted = model.predict(X_test)
    expected = y_test 

    # return classification report
    report = classification_report(expected, predicted, output_dict=True)
    return report


#Creates a classification report searborn heatmap for KNN clustering with k=3 and 845 samples.

#change sample_size and k inputs to your disgretion
report = knn_report(sample_size=0.005, k=3)

print(report)
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
plt.rcParams['figure.dpi'] = 3000
plt.rcParams['savefig.dpi'] = 3000
sns.color_palette("vlag", as_cmap=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="RdYlGn", ax=ax)
#plt.title("KNN Clustering (n=3)\nSample size: 845\n", {'fontsize':15, 'fontweight':"bold"})
plt.show() 