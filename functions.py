import os
from typing import List, Dict, Any
import pandas as pd
import pickle

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def count_of_stops(timestamps: List[float], speeds: List[float], headings: List[float]) -> int:
    
    merge = pd.DataFrame(
    {'timestamps': timestamps,
     'speeds': speeds})
    
    #Transform timestamp to DatetimeIndex
    datetime_series = pd.to_datetime(merge['timestamps'], unit='s')
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df=merge.set_index(datetime_index)
    
    #Calculate 3 second rolling max of speed
    df['rollmean'] = df['speeds'].rolling('3s',min_periods=3).max()
    
    #Flag stops where rolling 3 second average of speed is <= .5 meters per second 
    df['event'] = df['rollmean'].apply(lambda x: 1 if x <= .5 else 0)
    events = df[df['event']==1].copy()
    events['time_diff'] = events['timestamps'].diff()
    
    # Find and drop events within 3 seconds of each other i.e. duplicate events
    indexNames = events[events['time_diff'] <3 ].index
    events.drop(indexNames , inplace=True)
    
    count = events['event'].sum(axis = 0) 
    
    return count

def count_of_turns(timestamps: List[float], speeds: List[float], headings: List[float]) -> int:
    
    # Find true change in degree heading
    diff = [(j-i + 180 + 360) % 360 - 180 for i, j in zip(headings[:-1], headings[1:])]
    
    #remove NaNs and adjust list length
    change = pd.Series(diff).fillna(0).tolist()
    change.insert(0,0)
    
    # Calculate cumulative change in direction, resetting to zero and 
    # creating a turn flag when the value reaches 60 degrees in either direction.
  
    maxvalue = 60
    minvalue = -60

    lastvalue = 0
    newcum = []
    event = []
    
    for i, j in enumerate(change):
        thisvalue = j + lastvalue
        flag = 0
        if thisvalue > maxvalue:
            thisvalue = 0
            flag = 1
        elif thisvalue < minvalue:
            thisvalue = 0
            flag = 1
        newcum.append(thisvalue)
        event.append(flag)
        lastvalue = thisvalue
    
    merge = pd.DataFrame(
    {'timestamps': timestamps,
     'headings': headings,
     'change': change,
     'cum_change': newcum,
     'event': event
    })
    
     # Transform timestamp to DatetimeIndex
    datetime_series = pd.to_datetime(merge['timestamps'], unit='s')
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df=merge.set_index(datetime_index)
    
    # Differentiate between left and right turns
    df['direction'] = df['change'].apply(lambda x: 'R' if x < 0 else 'L' if x > 0 else '-')
    
    turns = df[df['event']==1].copy()
    
    # Separate left and right turns
    right_turns = turns[turns['direction']=='R'].copy()
    left_turns = turns[turns['direction']=='L'].copy()
         
    # Find and drop events within 3 seconds of each other i.e. duplicate events
    right_turns['Delta'] = right_turns['timestamps'].diff()
    left_turns['Delta'] = left_turns['timestamps'].diff()
    
    R_index = right_turns[right_turns['Delta'] <=3 ].index
    L_index = left_turns[left_turns['Delta'] <=3 ].index
    
    right_turns.drop(R_index , inplace=True)
    left_turns.drop(L_index , inplace=True)
    
    # Find final count
    r_count = right_turns['event'].sum(axis = 0) 
    l_count = left_turns['event'].sum(axis = 0) 
    
    count = r_count + l_count
    
    return count


def predict(features: Dict[str, Any]) -> int:
    with open(os.path.join(THIS_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
        prediction = model.predict(features)
    
    return prediction


def evaluate(y: List[int], y_hat: List[int]) -> Dict[str, float]:
    true_pos = sum(1 for x,y in zip(y,y_hat) if x==1 and y==1)
    true_neg = sum(1 for x,y in zip(y,y_hat) if x==0 and y==0)
    false_pos = sum(1 for x,y in zip(y,y_hat) if x==0 and y==1)
    false_neg = sum(1 for x,y in zip(y,y_hat) if x==1 and y==0)
    
    accuracy = (true_pos + true_neg) / len(y) 
    precision = true_pos / (true_pos + false_pos) 
    recall = true_pos / (true_pos + false_neg) 
    
    metrics = {"accuracy": accuracy, "recall": recall, "precision": precision}
    
    return metrics
