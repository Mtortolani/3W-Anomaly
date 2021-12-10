#import relevant modules

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

 
 
 
class ModelCreator():
    def __init__(self, processor):
        self.processor = processor
        self.total_data = processor.total_data
       
 
    #extract only sample of full data for testing, scrap data outside of sample
    def sampler(self, total_data, percent_kept):
        '''
        Dump part of the data so the ML model doesnt have to train on ENTIRE dataset
 
        Parameters:
        total_data (tuple): tuple of all X and Y values derived from DataProcessor.total_data_compiler
        percent_kept (float): what fraction of the entire data are we keeping
 
        Returns:
        (tuple): dataset tuple similar to total_data but smaller
        '''
        X, X_scrap, y, y_scrap = train_test_split(total_data[0], total_data[1], train_size=percent_kept, random_state=42)
        print(f"The length of the new dataset:", len(total_data[0]))
        return (X, y)
 
 
 
    def create_knn_model(self, sample_size=0.001,k=3):
        '''
        Run KNN classifier on data
       
        Parameters:
        sample_size (float): what fraction of the entire data are we keeping
        k (int): what K value should be used for the K Nearest Neighbor Classifier
       
        Returns:
        (dict): returns the classification report in dictionary format, reporting stats such as accurcy and recall
        '''
        #keep a portion of the data for the analysis, ignore the rest
        total_data_sample = self.sampler(self.total_data, sample_size)
 
        #split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(total_data_sample[0], total_data_sample[1], train_size=0.8,random_state=42)
 
        #knn model creation and fitting
        model = KNeighborsTimeSeriesClassifier(n_neighbors = k, metric="dtw")
        model.fit(X_train, y_train)
        print('Model Fitting Complete')
        predicted = model.predict(X_test)
        print('Model Predictions Complete')
        expected = y_test
 
        # return classification report in dictionary format
        report = classification_report(expected, predicted, output_dict=True)
        return report
 
 
 
    def knn_f1_score(self, sample_size=0.001, runs=10):
        '''
        Collects a list of f1 scores for knn clustering for different values of k (always odd)
       
        Parameters:
        sample_size (float): fraction of entire dataset to be used for model training
        runs (int): how many times the model should be trained at incrementing k values
        Returns:
        (list): list of f1 scores returned from models
        '''
 
        total_data_sample = self.sampler(self.total_data, sample_size)
       
        #split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(total_data_sample[0], total_data_sample[1], train_size=0.8,random_state=42)
 
        f1_list = []
        # run knn classifier at k=1,3,5,7,etc.
        for k in range(0, runs):
            model = KNeighborsTimeSeriesClassifier(n_neighbors = (1+2*k), metric="dtw")
            model.fit(X_train, y_train)
            print('Model Fitting Complete')
            predicted = model.predict(X_test)
            expected = y_test
 
            # find f1 score for each model and append to list
            score = f1_score(expected, predicted, average="weighted")
            f1_list.append(score)
        print(f1_list)
        return f1_list

