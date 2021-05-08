# Detecting Anomalies in Offshore Oil Wells

Members: Marco Tortolani

### Summary:
This project was designed to build off of the work provided by Vagas, Munaro, Ciarelli, Nedeuros, Amaral, Barrionuevo, Araujo, Ribeiro, Magalhaes in their paper A Realistic and Public Dataset with Rare and Undesirable Events in Oil Wells. From the data they collected on offshore oil well failures derived from 3 separate petrobras wells, I utilized K Nearest Neighbor Clustering with Dynamic Time Warping to create a model that can preemptively detect oil well failures. By splitting the multivariate time series dataset into minute long intervals to input into the KNN model, I was able to achieve an 85% Weighted F1 score utilizing 1% of the total data for training and testing. Further work would involve utilizing more efficient anomaly detention models such as isolation forest, utilizing cloud computing resources to work though greater portions of the data, and potentially work to to balance variance and bias if deemed necessary. Nonetheless, the result of this research concludes that it is feasible to create a model that can accurately target causes of well failures using multivariate time series sensor data with a moderate to high degree of certainty.

### Libraries utilized:
pandas, matplotlib, seaborn, os, statistics, sklearn, tslearn, random, numpy, scipy

### Notes to run code:
1. The dataset consists of nearly 2000 csv files which cannot easily be uploaded to github. Please upload the dataset from Kaggle as made available by user afrânio (Kaggle link in dataset citation). 
2. The functions loop_directory() and batch_dict_maker() assume that the 3W dataset is in the same file path as your working file, as it calls csv files solely from the well error number (0-8)  and measurement type (WELL, SIMULATED, DRAWN). If the functions are unable to find the appropriate files, considering altering how the files are called to directly reference your individual file path.
3. In its current state, the model has an exponential complexity in training time, so as you increase the train and test sizes to reflect greater portions of the data, it will eventually take hours/days to trains. As a reference, sample_size=0.001 takes minutes to run, meanwhile sample_size=0.01 takes roughly half a day. knn_report(sample_size=0.001) is written into the code, knn_report(sample_size=0.01) can be found in the included powerpoint presentation.


### Dataset Citation
Dataset uploaded to Kaggle by afrânio
Link: https://www.kaggle.com/afrniomelo/3w-dataset
  
  
'''
@article{VARGAS2019106223,
title = "A realistic and public dataset with rare undesirable real events in oil wells",
journal = "Journal of Petroleum Science and Engineering",
volume = "181",
pages = "106223",
year = "2019",
issn = "0920-4105",
doi = "https://doi.org/10.1016/j.petrol.2019.106223",
url = "http://www.sciencedirect.com/science/article/pii/S0920410519306357",
author = "Ricardo Emanuel Vaz Vargas and Celso José Munaro and Patrick Marques Ciarelli and André Gonçalves Medeiros and Bruno Guberfain do Amaral and Daniel Centurion Barrionuevo and Jean Carlos Dias de Araújo and Jorge Lins Ribeiro and Lucas Pierezan Magalhães",
keywords = "Fault detection and diagnosis, Oil well monitoring, Abnormal event management, Multivariate time series classification",
abstract = "Detection of undesirable events in oil and gas wells can help prevent production losses, environmental accidents, and human casualties and reduce maintenance costs. The scarcity of measurements in such processes is a drawback due to the low reliability of instrumentation in such hostile environments. Another issue is the absence of adequately structured data related to events that should be detected. To contribute to providing a priori knowledge about undesirable events for diagnostic algorithms in offshore naturally flowing wells, this work presents an original and valuable dataset with instances of eight types of undesirable events characterized by eight process variables. Many hours of expert work were required to validate historical instances and to produce simulated and hand-drawn instances that can be useful to distinguish normal and abnormal actual events under different operating conditions. The choices made during this dataset's preparation are described and justified, and specific benchmarks that practitioners and researchers can use together with the published dataset are defined. This work has resulted in two relevant contributions. A challenging public dataset that can be used as a benchmark for the development of (i) machine learning techniques related to inherent difficulties of actual data, and (ii) methods for specific tasks associated with detecting and diagnosing undesirable events in offshore naturally flowing oil and gas wells. The other contribution is the proposal of the defined benchmarks."
}
Vargas, Ricardo; Munaro, Celso; Ciarelli, Patrick; Medeiros, André; Amaral, Bruno; Barrionuevo, Daniel; Araújo, Jean; Ribeiro, Jorge; Magalhães, Lucas (2019), “Data for: A Realistic and Public Dataset with Rare Undesirable Real Events in Oil Wells”, Mendeley Data, v1. http://dx.doi.org/10.17632/r7774rwc7v.1 
'''
