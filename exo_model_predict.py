import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    precision_recall_curve,
)

class ExoTrainer:
    """
    Class that creates a model that predicts whether a system has more than one planet given a dataset that contains the following features: stellar mass, stellar temperature, stellar radius, and semi major axis of one planet
    
    Args:
        df (pd.dataframe): dataframe containing all relevant columns
        mass_col (str): name of column containing stellar mass
        temp_col (str): name of column containing stellar temperature
        rad_col (str): name of column containing stellar radius
        test_size (float): the proportion of the data to be saved for testing
    
    Returns:
        XGBClassifier: model to be used for predicting 
        pd.dataframe: dataset for model to be tested on
    
    """

    def __init__(self, df, mass_col, temp_col, rad_col, discmethod, test_size):
        self.df = df
        self.mass_col = mass_col
        self.temp_col = temp_col
        self.rad_col = rad_col
        self.discmethod = discmethod
        self.test_size = test_size

    def make_exomodel(self):
        
        """
        This function uses the data provided to it to train a model for predicting the number of planets in a given system. This model is based on the following features: stellar mass, stellar temperature, stellar radius, and semi major axis of one planet
        """
        
        # Turn the categorical discovery method variable into a numerical variable
        representation_map = {}
        for category in self.df[self.discmethod].unique():
            representation_map[category] = len(
                self.df[(self.df[self.discmethod] == category)]
            ) / len(self.df)
        self.df["pct_discmethod"] = self.df[self.discmethod].map(representation_map)

        
        # Turn the number of planets column into 0's for 1 planet and 1 for > 1 planet
        def multiple_planet_check(row):
            return 1 if row["pl_pnum"] > 1 else 0
        y = self.df.apply(multiple_planet_check, axis=1)
        
        # Select only the relevant columns to train on
        x = self.df[[self.mass_col, self.temp_col, self.rad_col, "pct_discmethod"]]

        # Split the dataset into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)
        
        # Create the model 
        model = XGBClassifier()
        model.fit(X_train, y_train)
        
        self.model = model
        
        return X_test, y_test, model
    
    def predict_exoplanets(self, X_test, y_test):
        """
        This function uses a model to predict the number of exoplanets that are within a given system on a test dataset. 
        
        Args:
            X_test (pd.dataframe): The data that the model should be tested on
            y_test (pd.Series): Column containing the labels that we're attempting to predict.
        """
        y_pred = self.model.predict(X_test)
        y_pred_labeled = ["multiple planets" if pred == 1 else "single planet" for pred in y_pred]
        y_pred_df = pd.DataFrame({"predictions":y_pred_labeled,"original_index":X_test.index.to_list()})
        y_pred_df.set_index("original_index", inplace=True)
 
        X_test["predictions"] = y_pred_df["predictions"]
    
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        precision_recall = precision_recall_curve(y_test, y_pred)
    
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Confusion:" + str(confusion * 100))
        print("Recall: %.2f%%" % (recall * 100.0))
        print("Precision: %.2f%%" % (precision* 100.0))

        return X_test, y_pred_df

