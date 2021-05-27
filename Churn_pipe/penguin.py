import pandas as pd
import numpy as np

class Processing ():
    def  __init__ (self, df):
        self.df = pd.read_csv(df)
        
    def Preprocessing (self, colname, target, value):
        df = self.df
        
        for i in colname:
            df[colname] = np.where(df[colname] == value, 1,0)
            
            df = df.select_dtypes(['int64', 'float64'])
            
            X = df.drop(target, axis = 'columns')
            y = df[target]
            
        return X, y