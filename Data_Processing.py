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
        # This is your directory where DataProcessor should look for data
        self.data_directory ='3W/' # loop_directory(folder)
        
        self.included_error_codes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # This variable if set to true will turn all error codes (0-8) into binary error codes (0 or 1)
        self.binary_error_detection = True
        # This variable encapsulates how many seconds of data you want the model to train/test on each time
        self.seconds = 60

        #Once all these variables are set up, you can run DataProcessor.clean_data()

        
        
        
    # purely internal function, does not need to be called
    # all data files downloaded from kaggle are in a folder called '3W', seperated into folders by error number ('0', '1', '2', etc)
    def loop_directory(self, folder: str, error_folder: str, type: str):
        '''
        Loop files in the directory of files by error type and oil well type to get dataframes
        
        Parameters:
        folder (str): name of folder where data is stored in
        error_folder (str): error code number (0-9) accessed as subdirectory
        type (str):l measurement type ("WELL", "DRAWN", "SIMULATED") to access files

        Returns:
        (list): returns list of every dataframe of a desired error code and measurement type

        Citation: Modified version of tutorial code by Khuyen Tran
        https://towardsdatascience.com/3-python-tricks-to-read-create-and-run-multiple-files-automatically-5221ebaad2ba
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
    def batch_dict_maker(self, main_data_dir='3W/', included_error_codes = [0,1,2,3,4,5,6,7,8], well_types=['WELL','DRAWN','SIMULATED'], nafill=True):
        '''
        Utilize loop_directory to turn csv files into a nested dictionary of oil sensor data, seperated by error type and well type.

        Parameters:
        main_data_dict (str): path to folder where data is stored
        included_error_codes (list): This is the list of error codes you wanna include in the model (0-8)
        nafill (bool): fill missing code values with NA
        well_types (list): Define where you want your measurements coming from (only real wells, hand drawn measurements, or simulated measurements)

        Returns:
        (dict): dictionary of dataframes organized by dict[error code][measurement type]
        
        '''
        df_dict = {}
        for error_code in included_error_codes:
            print(f"Creating dict: {error_code}")
            df_dict[error_code] = {}
            # Seperate csv files by the measurement methon (pulled from csv name)
            for type in well_types:
                df_dict[error_code][type] = self.loop_directory(main_data_dir, str(error_code), type)
                # for
                for df in df_dict[error_code][type]:
                    df['timestamp'] =  pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
                    # Drop unnecesarry columns
                    df.drop(columns = ["QGL"], inplace = True)
                    # OPTIONAL: decide wether error values are variable of binary
                    if self.binary_error_detection==True:
                        df["class"]=df["class"].astype(bool).astype(int)
                    # fill in NA's with zeros since tslearn does not accpect NA values
                    if nafill==True:
                        df.fillna(0, inplace=True)
        print("Dataframe dictionary making complete")
        return df_dict






    # Tslearn and sklearn like having its data into the X that we input into the model and the Y that we test for
    # Data from csv files is in intervals of ~daily data, which is too big for machine learning
    # we split up that large time series data into smaller time series, in this case 60 second intervals.

    def total_data_compiler(self, dataframe_dict, seconds=60, x_column_start=1, x_column_end=7):
        '''
        From nested dictionary of wells, creates a tuple of a list of x data values anda y list of class values for your model

        
        Parameters: nested dict, int, int, int
        dataframe_dict (dict): nested dictionary of all dataframes for model to use (organized by dict[error code][measurement type]) 
        seconds (int): how many seconds do you want your model to train on each time

        Returns:
        (tuple) tuple of 2 lists (X small time series inputs, Y class values)
        '''

        X = []
        y = []
        for error_number in dataframe_dict:
            for well_type in dataframe_dict[error_number]:
                print(f"Currently working on batch ({error_number},  {well_type})")
                for well in dataframe_dict[error_number][well_type]:
                    interval_amounts = len(well)//(seconds)
                    for interval in range(0, interval_amounts - 1):
                        # Cut out time interval
                        well_portion = well.iloc[seconds*interval : seconds*(interval+1)]
                        # Only get the columns we want (first six non-time columns)
                        X.append(well_portion.loc[:, x_column_start : x_column_end])
                        y.append(int(mode(well_portion["class"])))
        X = to_time_series_dataset(X)
        print("time series conversion complete")
        return (X, y)



    #total data is main time series dataset, contains entirety of data
    #ensuring that the time series has shape (n_ts number of time series, max_sz length of largest series, d dimensions of series)

    Processor = DataProcessor()
    Processor.total_data = total_data_compiler()
    print(Processor.total_data[0].shape)




    #total_data = ((60 seconds of time series data, what error type it is), (another 60 seconds, another error type), (etc))

    # intervals = [10, 30, 60, 120, 300]
    # total_data_time_list = []
    # for time_interval in intervals:
    #     interval_data = total_data_compiler(batch, seconds=time_interval)


    total_data[0].shape