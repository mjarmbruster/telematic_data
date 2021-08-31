# telematic_data
Feature Extraction Using Driver Telematic data

This goal of this project is to build a classification model using driver telematic data for an insurance company. It requires performing feaure extraction on the telematic time series of driver trips to identify which trips are actuarially interesting.

This repo contains:
􏰀 A dataset of N “trip” csv data files: timeseries data recorded from car trips including the vehicle speed and heading at different points in time
􏰀 A data matrix X ∈ RN×K of features about each trip
􏰀 A target vector y ∈ RN×1 indicating whether each trip is “interesting” or not
The objective of the work sample is to augment X by adding two new features xa1, xa2, and use the final feature set Xa = [X, xa1, xa2] ∈ RN×(K+2) to fit a model yˆ = f(Xa) to predict y on new data.

2 Data
The prepared files include:
􏰀 trip data.zip: a zipped folder containing the relevant csv trip data, where each trip has a series of records containing:
– time seconds: the time in seconds since the start of the trip (float) – speed meters per second: the speed in m/s of the vehicle (float
∈ [0, ∞), invalid values will be indicated with NaN)
– heading degrees: the angle in deg of the vehicle’s travel relative to north (clockwise positive, float ∈ [0,360), invalid values will be indicated with NaN)
1
􏰀 model data.csv: a csv file containing:
– a column filename to join to the file names in trip data.zip – the data matrix X as columns feature1, ..., featureN
– the vector y as column y


􏰀 xa1: the count of stops in the trip [int]
􏰀 xa2: the count of turns (greater than 60 degrees) [int]


where filename should correspond to the filenames in the trip data.zip folder, and prediction is a binary int (0 or 1)
To help with building the feature extraction, the trip data corresponding to filename = "0001.csv" should have approximately:
􏰀 1 stop
􏰀 9 turns
