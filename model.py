import numpy as np
import pandas as pd
# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# Models
import lightgbm as lgb

from sklearn.pipeline import Pipeline

class Model:


    def __init__(self):
        #self.mean_rate = None

        self.ct = ColumnTransformer(
        [("ohe1", OneHotEncoder(), ['transport_type']),
            ("ohe2", OneHotEncoder(), ['origin_kma']),
            ("ohe3", OneHotEncoder(), ['destination_kma']),
            #("ohe4", OneHotEncoder(), ['weekday']),
            ("ohe5", OneHotEncoder(), ['hour']),
            ("ohe6", OneHotEncoder(), ['quarter']),
            ("ohe7", OneHotEncoder(), ['year']),
            ('StSc', StandardScaler(), ['weight','valid_miles'])
            ])

        self.ft = FunctionTransformer(self.__base_transform)

        self.pipe = Pipeline(
            steps=[
            ('base_transform', self.ft),
            ('ColumnTransf', self.ct),
            ('LR', lgb.LGBMRegressor(num_leaves=3000))
                ],
             verbose=True)

    def fit(self, x, y):
        #self.mean_rate = y.mean()
        print("result Y shape:", len(y)) 
        x.drop(columns=['rate'], inplace=True)   
        self.pipe.fit(x,y)
        return self

    def predict(self, x):
        return self.pipe.predict(x)

    def __base_transform(self, X_raw_df):
        X_raw_df.drop(columns=['rate'], inplace=True, errors='ignore')
        #X_raw_df.drop(columns=['weight', 'valid_miles'], inplace=True, errors='ignore')
        X_raw_df.fillna(method="backfill", inplace=True)
        X_raw_df['pickup_date'] = pd.to_datetime(X_raw_df['pickup_date'])
        
        X_raw_df['hour'] = X_raw_df.pickup_date.dt.hour
        X_raw_df['month'] = X_raw_df.pickup_date.dt.month
        X_raw_df['dayofyear'] = X_raw_df.pickup_date.dt.dayofyear
        X_raw_df['weekday'] = X_raw_df.pickup_date.dt.weekday
        X_raw_df['year'] = X_raw_df.pickup_date.dt.year
        X_raw_df['quarter'] = X_raw_df.pickup_date.dt.quarter
        X_raw_df.set_index('pickup_date', inplace=True)

        X_raw_df["valid_miles"] = X_raw_df["valid_miles"].apply(lambda x: np.log(x))
        #print("before X shape:", X_raw_df.shape)
        return X_raw_df
