# Detecting Anomalies in Offshore Oil Wells

Members: Marco Tortolani

### Summary:
Utilized KNN Clustering with DTW model to preemtively detect oil well failure. Results in an 85% weighted F1 score when utilizing 1% of the total data for training and testing.

### Description:
This project was designed to build off of the work provided by Vagas, Munaro, Ciarelli, Nedeuros, Amaral, Barrionuevo, Araujo, Ribeiro, Magalhaes in their paper A Realistic and Public Dataset with Rare and Undesirable Events in Oil Wells. From the data they collected on offshore oil well failures derived from 3 separate petrobras wells, I utilized K Nearest Neighbor Clustering with Dynamic Time Warping to create a model that can preemptively detect oil well failures. By splitting the multivariate time series dataset into minute long intervals to input into the KNN model, I was able to achieve an 85% Weighted F1 score utilizing 1% of the total data for training and testing. Further work would involve utilizing more efficient anomaly detention models such as isolation forest, utilizing cloud computing resources to work though greater portions of the data, and potentially work to to balance variance and bias if deemed necessary. Nonetheless, the result of this research concludes that it is feasible to create a model that can accurately target causes of well failures using multivariate time series sensor data with a moderate to high degree of certainty.
