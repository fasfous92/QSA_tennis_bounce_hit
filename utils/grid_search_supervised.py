import json
import pandas as pd
import numpy as np
from preprocess import prepare_data
from model_supervised import ActionClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb # standard import for the grid search

def perform_grid_search(X, y):
    """
    Performs a Grid Search using standard LightGBM + GridSearchCV.
    Compares the 'Default' ActionClassifier vs. the 'Tuned' LightGBM parameters.
    Saves confusion matrices and prints reports for comparison.
    """
    print("\n" + "="*50)
    print("STARTING GRID SEARCH (Inspired by StackOverflow Example)")
    print("="*50)
    classes = np.unique(y)
    # 1. Create a hold-out set for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # BEFORE: Evaluate Default Model (Your Wrapper)
    # ---------------------------------------------------------
    print("\n--- 1. Evaluating Default ActionClassifier (Before) ---")
    model_default = ActionClassifier()
    # Note: We pass the specific split to train, assuming .train() handles internal logic
    # If ActionClassifier.train() splits internally, we pass X_train. 
    model_default.train(X_train, y_train, test_size=0.2, val_size=0.2)
    
    y_pred_default = model_default.predict(X_test)
    print("\n[Default Model] Classification Report:")
    print(classification_report(y_test, y_pred_default))

    # Save Confusion Matrix (Before)
    cm_default = confusion_matrix(y_test, y_pred_default, labels=model_default.classes_)
    #cm_default = confusion_matrix(y_test, y_pred_default)
    disp_default = ConfusionMatrixDisplay(confusion_matrix=cm_default,display_labels=model_default.classes_)
    disp_default.plot(cmap='Blues')
    plt.title("Confusion Matrix: Default Model")
    plt.savefig('./img/confusion_matrix_before.png')
    print("Saved './img/confusion_matrix_before.png'")

    # ---------------------------------------------------------
    # GRID SEARCH: Using Standard LGBMClassifier + GridSearchCV
    # ---------------------------------------------------------
    print("\n--- 2. Running GridSearchCV ---")
    
    # Define the estimator
    estimator = lgb.LGBMClassifier(verbose=-1)
    
    # Define the grid (StackOverflow inspired structure)
    param_grid = {
        'learning_rate': [0.01,0.1],
        'n_estimators': [50, 100],
        'num_leaves': [20,],
        'max_depth': [-1, 10]
    }
    
    # Initialize GridSearchCV
    gsearch = GridSearchCV(estimator, param_grid, cv=3, scoring='accuracy', verbose=1)
    gsearch.fit(X_train, y_train)

    best_params = gsearch.best_params_
    best_score = gsearch.best_score_
    
    print(f"\nBest Parameters found: {best_params}")
    print(f"Best CV Score: {best_score:.4f}")

    # ---------------------------------------------------------
    # AFTER: Evaluate Optimized Model
    # ---------------------------------------------------------
    print("\n--- 3. Evaluating Optimized Model (After) ---")
    
    # We use the best_estimator_ directly from GridSearchCV for prediction
    best_model = gsearch.best_estimator_
    y_pred_opt = best_model.predict(X_test)

    print("\n[Optimized Model] Classification Report:")
    print(classification_report(y_test, y_pred_opt))

    # Save Confusion Matrix (After)
    cm_opt = confusion_matrix(y_test, y_pred_opt, labels=best_model._classes)
    #cm_opt = confusion_matrix(y_test, y_pred_opt)
    disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt,display_labels=best_model._classes)
    disp_opt.plot(cmap='Greens')
    plt.title(f"Confusion Matrix: Optimized")
    plt.savefig('./img/confusion_matrix_after.png')
    print("Saved './img/confusion_matrix_after.png'")
    
    # Optional: Save the best model
    best_model.booster_.save_model('./models/supervised_model_optimized.txt')
    print("Saved './models/supervised_model_optimized.txt'")


if __name__ == "__main__":
    
    #treats all the json files seperatly and return the full dataset
    #to be split into train,validation,test
    full_df=prepare_data()
    
    #split data
    X=full_df.drop(columns=['action']).astype(float)
    y=full_df['action']
    
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # #intialize model Lgbm
    # lgbm=ActionClassifier()
    
    # #train model
    # #This method already handles data splitting into train,val,test
    # lgbm.train(X,y,test_size=0.2,val_size=0.2)
    
    # #evaluate model
    # lgbm.evaluate()
    
    # #save model
    # lgbm.save_model(filename='./models/my_catboost_model.cbm')

    # --- ADDED METHOD CALL ---
    perform_grid_search(X, y)
