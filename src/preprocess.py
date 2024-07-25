import pandas as pd
import numpy as np
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from features import convert_to_datetime , haversine, Extract_Datetimeinfo ,Discretisation
from sklearn.preprocessing import OneHotEncoder

def get_data(path):
    return pd.read_csv(path/"train.csv")

def preprocess_pipeline(df):
    
    date_time_transf = ColumnTransformer([("Datetime_tranf",convert_to_datetime(),["pickup_datetime","dropoff_datetime"])],remainder="passthrough",verbose_feature_names_out=False)
    date_time_transf.set_output(transform="pandas")

    pipeline1 = Pipeline([("date_time",date_time_transf),
                        ("haversine",haversine()),
                        ("Datetime_info",Extract_Datetimeinfo()),
                        ("hours_binning",Discretisation())
                        ])
    
    df = pipeline1.fit_transform(df)
    

    # removing outliers based on speed
    invalid_indexes = df[df["speed"] >= 100].index
    df = df.drop(index=invalid_indexes)
    
    invalid_indexes = df[df["speed"] < 4].index
    df = df.drop(index=invalid_indexes)

    # removing outliers based on distance
    invalid_indexes = df[df["distance"] <= 0.500].index
    df = df.drop(index=invalid_indexes)

    # removing outliers based on time_duration
    invalid_indexes = df[df["trip_duration"] <= (2/60)].index
    df = df.drop(index=invalid_indexes)



    labels = ["Midnight","Early Morning","Morning","Late Morning","Afternoon","Late Afternoon","Evening","Night"]
    
    feature_enc = ColumnTransformer([("hrs_enc",OrdinalEncoder(categories=[labels]),["hours"]) ,
                                ("day_enc",OneHotEncoder(handle_unknown='ignore',sparse_output=False),["day"]),
                                ("store_and_fwd_flag",OrdinalEncoder(),["store_and_fwd_flag"])
                                ],remainder="passthrough",verbose_feature_names_out=False)

    
    feature_enc.set_output(transform="pandas")

    pipeline2 = Pipeline([
        ("pipeline1",pipeline1),
        ("time_enc",feature_enc)
    ])

    df = pipeline2.fit_transform(df)  
    
    return df

def save_data(df,path):
    pathlib.Path(path).mkdir(exist_ok=True,parents=True)
    df.to_csv(path/"preprocessed.csv",index=False)




def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent
    data_path = home_dir / "data" / "raw"
    save_path = home_dir / "data" / "interim"

    df = get_data(data_path)
    df = preprocess_pipeline(df)
    save_data(df,save_path)


if __name__ == "__main__" :
    main()