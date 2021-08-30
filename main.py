import os
import sys
import functions
import pandas as pd

THIS_DIR = os.path.abspath('')

def main(path_to_model_data: str, path_to_trip_data: str) -> None:
    model_data = pd.read_csv(path_to_model_data)
    filenames = []
    stops = []
    turns = []
    os.chdir(path_to_trip_data)
    for file in os.listdir(path_to_trip_data):
        if file.endswith(".csv"):
            trips = pd.read_csv(file)
            timestamps = trips.time_seconds.tolist()
            speeds = trips.speed_meters_per_second.tolist()
            headings = trips.heading_degrees.tolist()
            stop_count = functions.count_of_stops(timestamps, speeds, headings)
            turn_count = functions.count_of_turns(timestamps, speeds, headings)
            
            filenames.append(file)
            stops.append(stop_count)
            turns.append(turn_count)
            
    os.chdir(THIS_DIR)
            
    add_features = pd.DataFrame(list(zip(filenames, stops, turns)), 
               columns =['filename','stops','turns']) 
    df = pd.merge(model_data, add_features, how='left', on='filename', 
                  validate='one_to_one')

    predictors = [x for x in df.columns if x not in ['y','filename']]
    train = df[predictors]
    y = df['y']
    
    y_hat = functions.predict(train)
    
    pred = pd.DataFrame(list(zip(filenames,y_hat)), columns=['filename','prediction']) 
    pred.to_csv(sys.stdout, index=False) 
    
    
    return


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise TypeError('Please pass two args as strings')
    if not (type(sys.argv[1]) is str and type(sys.argv[2]) is str):
        raise TypeError('Please pass two args as strings')
    main(sys.argv[1], sys.argv[2])
