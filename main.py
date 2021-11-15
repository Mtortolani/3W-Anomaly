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
 
from Data_Processing import DataProcessor
from Model_Creation import ModelCreator
 
# Initialize the Data processor and
Processor = DataProcessor()
Processor.total_data_compiler()
print(Processor.total_data[0].shape)
 
 
#Creates a classification report searborn heatmap for KNN clustering with k=3 and 845 samples.
#change sample_size and k inputs to your disgretion
ModelCreator(Processor)
report = ModelCreator.knn_report(sample_size=0.005, k=3)
 
#Graph the report into a heatmap
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
plt.rcParams['figure.dpi'] = 3000
plt.rcParams['savefig.dpi'] = 3000
sns.color_palette("vlag", as_cmap=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="RdYlGn", ax=ax)
plt.title("KNN Clustering (n=3)\nSample size: 845\n", {'fontsize':15, 'fontweight':"bold"})
plt.show()
 
 
scores = ModelCreator.knn_f1_score(sample_size=0.001, runs=25)
x = [1 + 2*i for i in range(25)]
 
#plot k vs f1 scores into a bar graph
sns.set_theme(style="whitegrid")
sns.barplot(x, scores, color="b")
plt.rcParams['figure.dpi'] = 3000
plt.rcParams['savefig.dpi'] = 3000
plt.xlabel("K Neighbors Considered")
plt.ylabel("F1 Score")
sns.despine()