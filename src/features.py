import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class convert_to_datetime(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass
    def fit(self,X):
        return self
    def transform(self,X):
        X = X.copy()

        for col in X.columns:
            X[col] = pd.to_datetime(X[col])
        
        return X
    
    def set_output(self,transform="default"):
        pass

class haversine(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X):
        return self
    def transform(self,df):
        df=df.copy()
        # Convert decimal degrees to radians
        lon1 = np.radians(df["pickup_longitude"].values)
        lat1 = np.radians(df["pickup_latitude"].values) 
        lon2 = np.radians(df["dropoff_longitude"].values) 
        lat2 = np.radians(df["dropoff_latitude"].values)

        # Haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        r = 6371  # Radius of Earth in kilometers. Use 3956 for miles.
        df["distance"] =  np.round((c * r),2)
        df["trip_duration"]= df["trip_duration"] / 3600 #converting time duartion in hours 
        df["speed"] = df["distance"]/df["trip_duration"] # speed represented in km/hr
        return df
    
    def set_output(self,transform="default"):
        pass

class Extract_Datetimeinfo(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X):
        return self
    def transform(self,df):
        df=df.copy()
        df["month"] = df["pickup_datetime"].dt.month
        df["hours"] = df["pickup_datetime"].dt.hour
        df['day'] = df['pickup_datetime'].dt.day_name()
        return df
    
    def set_output(self,transform="default"):
        pass

class Discretisation(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X):
        return self
    def transform(self,df):
        bins = [0,2,6,9,12,15,18,21,24]
        labels = ["Midnight","Early Morning","Morning","Late Morning","Afternoon","Late Afternoon","Evening","Night"]
        df["hours"] = pd.cut(df["hours"].values,bins,labels=labels,right=False)
        return df
    
    def set_output(self,transform="default"):
        pass





