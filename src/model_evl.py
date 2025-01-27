import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score
import json

def load_data(test_path):
    test_df=pd.read_csv(test_path)
    return test_df

def x_y_values(test_df):
    x_test=test_df.iloc[:,:-1].values
    y_test=test_df.iloc[:,-1].values
    return x_test,y_test

def score(model_name,x_test,y_test):
    model=joblib.load(model_name)
    y_pred=model.predict(x_test)

    mitrics_dic={
        'accuracy':accuracy_score(y_pred,y_test),
        'recall':recall_score(y_pred,y_test),
        'precision':precision_score(y_pred,y_test)
    }

    with open('metrics.json','w') as file:
        json.dump(mitrics_dic,file,indent=3)
    return mitrics_dic

test_path='./data/pro/test_pre.csv'

def main():
    test_df=load_data(test_path)
    x_test,y_test=x_y_values(test_df)
    score('model.pkl',x_test,y_test)

if __name__=='__main__':
    main()