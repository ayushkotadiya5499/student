import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(url):
    df=pd.read_csv(url)
    return df

def train_test(df,test_size):
    train_df,test_df=train_test_split(df,test_size=test_size,random_state=5)
    return train_df,test_df

def save_data(data_path,train_df,test_df):
    os.makedirs(data_path)
    train_df.to_csv(os.path.join(data_path,'train.csv'))
    test_df.to_csv(os.path.join(data_path,'test.csv'))

def main():
    df=load_data('C:/Users/ayush/Downloads/diabetes.csv')
    train_df,test_df=train_test(df,0.2)
    data_path='data/raw'
    save_data(data_path,train_df,test_df)

if __name__=='__main__':
    main()