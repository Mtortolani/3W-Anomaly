#import relevant modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

 
from Data_Processing import DataProcessor
from Model_Creation import ModelCreator


def main():
    # Initialize the Data processor and prepare Processor.total_data
    Processor = DataProcessor()
   
    # This is your directory where DataProcessor should look for data
    main_data_dir = 'data/'
    # These are the error codes you want to test for
    included_error_codes = [0,1,2,3,4,5,6,7,8]
    # You can choose which measurement types to include
    well_types=['WELL','DRAWN','SIMULATED']
    # This variable if set to true will turn all error codes (0-8) into binary error codes (0 or 1)
    binary_error_detection = False
    # Wether you want to impute missing values (Leave true unless you implement an alternate imputation method)
    nafill = True
    
    data_dict = Processor.batch_dict_maker(main_data_dir, included_error_codes, well_types,  binary_error_detection, nafill)

    # How many seconds of data you want the model to use for each train/test sample
    seconds=60
    # Which error columns from the oil dataframes do you want to include
    x_column_start=1
    x_column_end=7
    
    Processor.total_data = Processor.total_data_compiler(data_dict, seconds, x_column_start, x_column_end)
    print(f'The internal data used is of shape: {Processor.total_data[0].shape}')
    
    #Initialize model creator object
    Creator = ModelCreator(Processor)
    
    # Creates a classification report searborn heatmap for KNN clustering with k=3 and 845 samples.
    # change sample_size and k inputs to your disgretion
    
    report = Creator.create_knn_model(sample_size=0.001, k=3)
    
    # Plot the report into a heatmap
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="RdYlGn", ax=ax)
    plt.title("KNN Clustering (n=3)\nSample size: 845\n", {'fontsize':15, 'fontweight':"bold"})
    plt.show()
    
    # How many k values do you want to compare the performance of (ie. runs=5 -> k=1,3,5,7,9)
    runs = 10
    # What portion of the entire dataset do you want to feed into model (note: time to run grows exponentially with data inputed)
    sample_size=0.001
    scores = Creator.knn_f1_score(sample_size, runs)
    k_values = [1 + 2*i for i in range(runs)]
    
    # Plot k vs f1 scores into a bar graph
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=k_values, y=scores)
    ax.set(xlabel="K Neighbors Considered", ylabel = "F1 Score")
    sns.despine()
    plt.show()




if __name__=='__main__':
    main()
    
    
    # Note when pushing to github: Use this command: [git push origin HEAD:main]