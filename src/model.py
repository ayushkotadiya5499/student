import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def load_data(train_path):
    train_df=pd.read_csv(train_path)
    return train_df

def x_y_values(train_df):
    x_train=train_df.iloc[:,:-1].values
    y_train=train_df.iloc[:,-1].values
    return x_train,y_train

def model_fit(model,x_train,y_train):
    model.fit(x_train,y_train)

def save_model(model):
    return joblib.dump(model,'model.pkl')

train_path='./data/pro/train_pre.csv'

def main():
    train_df=load_data(train_path)
    x_train,y_train=x_y_values(train_df)
    model=XGBClassifier()
    model_fit(model,x_train,y_train)
    save_model(model)

if __name__=='__main__':
    main()