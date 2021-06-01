import pandas as pd
import numpy as np

class Processing ():
   
    def  __init__ (self, df):
        self.df = pd.read_csv(df)
        
        
    def Preprocessing (self, colname, target, value):
        
        df = self.df
        
        for col in colname:
            df[col] = np.where(df[col] == value, 1,0)
            
        df = df.select_dtypes(include = ['int64', 'float64', 'int32'])
            
        X = df.drop(target, axis = 1)
        y = df[target]
            
        return X, y