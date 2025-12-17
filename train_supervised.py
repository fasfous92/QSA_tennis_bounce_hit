import json
import pandas as pd
import numpy as np
from utils.preprocess import prepare_data
from utils.model_supervised import ActionClassifier
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    #treats all the json files seperatly and return the full dataset
    #to be split into train,validation,test
    full_df=prepare_data()
    
    #split data
    X=full_df.drop(columns=['action'])
    y=full_df['action']
    
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    #intialize model Lgbm
    lgbm=ActionClassifier()
    
    #train model
    #This method already handles data splitting into train,val,test
    lgbm.train(X,y,test_size=0.2,val_size=0.2)
    
    #evaluate model
    lgbm.evaluate()
    
    #save model
    lgbm.save_model(filename='./models/my_catboost_model.cbm')
    
    
