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




class DataProcessor():
    def __init__(self):
        self.total_data = ()
        
        
        
        
    # all data files downloaded from kaggle are in a folder called '3W', seperated into folders by error number ('0', '1', '2', etc)
    def loop_directory(folder: str, error_folder: str, type: str):
        '''
        Loop files in the directory of files by oil well type ("WELL", "DRAWN", "SIMULATED.)
        Citation: Modified version of tutorial code by Khuyen Tran
        https://towardsdatascience.com/3-python-tricks-to-read-create-and-run-multiple-files-automatically-5221ebaad2ba
        
        Inputs:
        folder <- name of folder where data is stored in
        error_folder <- str() of error code number (0-9) accessed as subdirectory
        type <- well measurement type ("WELL", "DRAWN", "SIMULATED") to access files
        '''
        
        dir_list = []
        type = type.upper()
        # Open main data folder, look through each error sub-folder
        for filename in os.listdir(folder+error_folder):
            if filename.startswith(type):
                file_directory = os.path.join(folder+error_folder, filename)
                file_read = pd.read_csv(file_directory)
                dir_list.append(file_read)
        return dir_list



    # turns list of csv files into dictionary so we can divide data by error code or well type
    def batch_dict_maker(batch_code_start=0, batch_code_end=8, class_bool=True, nafill=True):
        '''
        Utilize loop_directory to turn csv files into a nested dictionary of oil sensor data, seperated by error type and well type.
        Note: class_bool dictates wether the class column will report the error type (False) or simply wether there was an error (True) as 0-1
        Input: int(0-8), int(0-8), bool, bool
        Output: dict
        '''
        main_data_dir = '3W/'
        batch = {}
        for batch_number in range(batch_code_start,batch_code_end+1):
            print(f"Creating batch: {batch_number}")
            batch[batch_number] = {}
            # Seperate csv files by the measurement methon (pulled from csv name)
            for type in ["WELL", "DRAWN", "SIMULATED"]:
                batch[batch_number][type] = loop_directory(main_data_dir, str(batch_number), type)
                # for
                for df in batch[batch_number][type]:
                    df['timestamp'] =  pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
                    # Drop unnecesarry columns
                    df.drop(columns = ["QGL"], inplace = True)
                    # OPTIONAL: decide wether error values are variable of binary
                    if class_bool==True:
                        df["class"]=df["class"].astype(bool).astype(int)
                    # fill in NA's with zeros since tslearn does not accpect NA values
                    if nafill==True:
                        df.fillna(0, inplace=True)
        print("Batch making complete")
        return batch

    batch = batch_dict_maker(class_bool=False)




    # Tslearn and sklearn like having its data into the X that we input into the model and the Y that we test for
    # Data from csv files is in intervals of ~daily data, which is too big for machine learning
    # we split up that large time series data into smaller time series, in this case 60 second intervals.

    def total_data_compiler(batch=batch, seconds=60, x_column_start=1, x_column_end=7):
        '''
        From nested dictionary of wells, creates a tuple of two lists, one being a time series dataset and another being a list of class values
        Note: use parameters to edit length of time series input for model and which variable columns are considered
        Input: nested dict, int, int, int
        Output: tuple of 2 lists (time series dataset, class values)
        '''
        X = []
        y = []
        for batch_number in batch:
            for well_type in batch[batch_number]:
                print(f"Currently working on batch ({batch_number},  {well_type})")
                for well in batch[batch_number][well_type]:
                    interval_amounts = len(well)//(seconds)
                    for interval in range(0, interval_amounts - 1):
                        well_portion = well.iloc[seconds*interval : seconds*(interval+1)]
                        X.append(well_portion.iloc[:, x_column_start : x_column_end])
                        y.append(int(mode(well_portion["class"])))
        X = to_time_series_dataset(X)
        print("time series conversion complete")
        return (X, y)


    self.total_data = total_data_compiler(batch, seconds=60)



    #total_data = ((60 seconds of time series data, what error type it is), (another 60 seconds, another error type), (etc))

    # intervals = [10, 30, 60, 120, 300]
    # total_data_time_list = []
    # for time_interval in intervals:
    #     interval_data = total_data_compiler(batch, seconds=time_interval)


    #total data is main time series dataset, contains entirety of data
    #ensuring that the time series has shape (n_ts number of time series, max_sz length of largest series, d dimensions of series)
    total_data[0].shape