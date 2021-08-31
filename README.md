# telematic_data
Feature Extraction Using Driver Telematic data

This goal of this project is to build a classification model using driver telematic data for an insurance company. It requires performing feaure extraction on the telematic time series of driver trips to identify which trips are actuarially interesting.

Project files:
trip data.zip: a zipped folder containing the relevant csv trip data, where each trip has a series of records containing timeseries data recorded from car trips including the vehicle speed and heading at different points in time and the heading in degrees (i.e. the angle of the vehicle’s travel relative to north (clockwise positive)

model data.csv: a csv file containing: a column filename to join to the file names in trip data.zip – the data matrix X containing other features to be used in classification.

The goal is to use the trip data to extract:

- xa1: the count of stops in the trip [int]
- xa2: the count of turns (greater than 60 degrees) [int]

Solution files:

