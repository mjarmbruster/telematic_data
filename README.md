# telematic_data
Feature Extraction Using Driver Telematic data

This goal of this project is to build a classification model using driver telematic data for an insurance company. It requires performing feaure extraction on the telematic time series of driver trips to identify which trips are actuarially interesting.

Project files:
trip data.zip: a zipped folder containing the relevant csv trip data, where each trip has a series of records containing timeseries data recorded from car trips including the vehicle speed and heading at different points in time and the heading in degrees (i.e. the angle of the vehicle’s travel relative to north (clockwise positive)

model data.csv: a csv file containing: a column filename to join to the file names in trip data.zip – the data matrix X containing other features to be used in classification.

The goal is to use the trip data to extract:

- xa1: the count of stops in the trip
- xa2: the count of turns (greater than 60 degrees)

Solution files:

main.py: the main file to be run on the command line.

Functions.py: A series of functions created for feature extraction and model evaluation.

model_building.ipynb: A notebook cotaining model training, hyperparameter tuning and validation.

Results:

Feature Extraction Functions

Count of stops:

I define a stop in the function to have occurred when the maximum speed over a 3 second rolling window is less than or equal to 0.5 meters per second. This seemed like a reasonable threshold given that in the real world, drivers may functionally stop without completely stopping all forward movement. The rolling window is designed to satisfy the guideline to exclude events lasting less than 3 seconds. Once stop events are flagged, if two events occur within 3 seconds of one another, the second event is dropped.

Count of turns:

This was the more difficult to build of the two feature extraction functions. The first challenge was normalizing the true change in direction when a driver’s heading crossed the 0/360 threshold. I ultimately solved this problem using modulo arithmetic. The second obstacle was calculating a cumulative sum of the car’s change in heading that would reset to zero each time a turn was made (change of +/- 60 degrees). My solution in this case was to create a For Loop combined with conditional statements to reset the incrementing variable.

Model Performance:

Based on initial performance results, I ultimately decided to use a gradient boosting model. After feature selection and hyperparameter tuning (using cross validation grid search), the final model attained the following performance metrics on out-of-sample data:
Accuracy: 0.87 Recall: 0.76 Precision: 0.79


