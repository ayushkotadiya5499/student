import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import  StandardScaler

def load_data(train_path,test_path):
    train_df=pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)
    return train_df,test_df


def x_y_values(train_df,test_df):
    x_train=train_df.iloc[:,:-1].values
    y_train=train_df.iloc[:,-1].values
    x_test=test_df.iloc[:,:-1].values
    y_test=test_df.iloc[:,-1].values
    return x_train,y_train,x_test,y_test

def preprocessing(s,x_train,x_test):
    x_train_pre=s.fit_transform(x_train)
    x_test_pre=s.transform(x_test)
    return x_train_pre,x_test_pre

def combine(x_train_pre,y_train,x_test_pre,y_test):
    x_train_pre=pd.DataFrame(x_train_pre)
    x_test_pre=pd.DataFrame(x_test_pre)
    y_train=pd.DataFrame(y_train,columns=['target'])
    y_test=pd.DataFrame(y_test,columns=['target'])

    train_pre=pd.concat([x_train_pre,y_train],axis=1,ignore_index=False)
    test_pre=pd.concat([x_test_pre,y_test],axis=1,ignore_index=False)
    return train_pre,test_pre

def save_data(data_path,train_pre,test_pre):
    os.makedirs(data_path)
    train_pre.to_csv(os.path.join(data_path,'train_pre.csv'))
    test_pre.to_csv(os.path.join(data_path,'test_pre.csv'))

train_path='./data/raw/train.csv'
test_path='./data/raw/test.csv'

def main():
    train_df,test_df=load_data(train_path,test_path)
    x_train,y_train,x_test,y_test=x_y_values(train_df,test_df)
    x_train_pre,x_test_pre=preprocessing(StandardScaler(),x_train,x_test)
    train_pre,test_pre=combine(x_train_pre,y_train,x_test_pre,y_test)
    data_path='data/pro'
    save_data(data_path,train_pre,test_pre)

if __name__=='__main__':
    main()