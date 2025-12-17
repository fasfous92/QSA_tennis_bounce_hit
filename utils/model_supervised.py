import catboost as ctb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

class ActionClassifier:
    def __init__(self, iterations=500, learning_rate=0.05, random_seed=42):
        """
        Initializes the CatBoostClassifier with specific hyperparameters.
        """
        self.model = ctb.CatBoostClassifier(
            iterations=iterations,               
            learning_rate=learning_rate,
            loss_function='MultiClass',  
            random_seed=random_seed,
            verbose=100,                 
            allow_writing_files=False,
        )
        self.X_test = None
        self.y_test = None
        self.classes_ = None

    def train(self, X, y, test_size=0.2, val_size=0.2):
  
        print("\n--- Splitting Data ---")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Split the remaining data into Train and Validation

        relative_val_size = val_size / (1.0 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=42
        )
        
        # Store test data
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"Train samples: {len(X_train)}")
        print(f"Valid samples: {len(X_val)}")
        print(f"Test samples:  {len(X_test)}")

        print("\n--- Starting CatBoost model training ---")
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),  
            early_stopping_rounds=20,
        )
        
        self.classes_ = self.model.classes_
        print("Training finished.")

    def evaluate(self):
        """
        Evaluates the trained model on the held-out Test set.
        """
        if self.X_test is None:
            print("Error: Model has not been trained or test data is missing.")
            return

        print("\n--- 3. Evaluation and Prediction ---")
        
        # Predict on the hidden test set
        y_pred = self.model.predict(self.X_test).flatten()

        # 1. Accuracy
        acc = accuracy_score(self.y_test, y_pred)
        print(f"\nOverall Accuracy: {acc:.4f}")

        # 2. Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # 3. Confusion Matrix Plot
        print("\Saving Confusion Matrix...")
        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes_)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix - Test Set")
        plt.savefig("others/confusion_matrix_lgbm.png", dpi=300, bbox_inches='tight')

    def save_model(self, filename='my_catboost_model.cbm'):
        """Saves the trained model to a file."""
        self.model.save_model(filename)
        print(f"\nModel saved to {filename}")

    def load_model(self, filename):
        """Loads a model from a file."""
        self.model.load_model(filename)
        # Update classes_ attribute after loading
        self.classes_ = self.model.classes_
        print(f"Model loaded from {filename}")
        
    def predict(self, X):
        """
        Predicts class labels for new input data X.
        
        Args:
            X: Features (DataFrame or numpy array) matching the training structure.
        
        Returns:
            numpy array of predicted class labels (e.g., ['air', 'bounce', ...])
        """
        # .flatten() is used because CatBoost sometimes returns an array of arrays
        return self.model.predict(X).flatten()