import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle as pkl

class Predictor:
    """
    Class that predicts whether a system has more than one planet given a dataset that contains the following features: stellar mass, stellar temperature, stellar radius, and semi major axis of one planet
    
    Args:
        df (pd.ataframe): dataframe containing all relevant columns
        mass_col (str): name of column containing stellar mass
        temp_col (str): name of column containing stellar temperature
        rad_col (str): name of column containing stellar radius
        sma_col (str): name of column containing the semi major axis of one planet
    """

    def __init__(self, model):
        self.model = model

    def load_data(self, df, mass_col, temp_col, rad_col, sma_col, discmethod):
        self.df = df
        self.mass_col = mass_col
        self.temp_col = temp_col
        self.rad_col = rad_col
        self.sma_col = sma_col
        self.discmethod = discmethod

    def make_prediction(self):
        df = self.df.rename(
            {
                self.mass_col: "st_mass",
                self.temp_col: "st_teff",
                self.rad_col: "st_rad",
                self.sma_col: "pl_orbsmax",
            }
        )

        representation_map = {}
        for category in df[self.discmethod].unique():
            representation_map[category] = len(
                df[(df[self.discmethod] == category)]
            ) / len(df)
        df["pct_discmethod"] = df[self.discmethod].map(representation_map)

        df.drop(self.discmethod, axis=1)

        df = df[["st_mass", "st_teff", "st_rad", "pl_orbsmax", "pct_discmethod"]]

        y_pred = self.model.predict(df)
        df["predictions"] = pd.Series(
            [
                "Likely more than 1 planet"
                if round(value) == 1
                else "Likely only 1 planet"
                for value in y_pred
            ]
        ).values

        self.prediction_df = df

        return df

model = pkl.load(open("num_planets_model.p", "rb"))

def create_num_planets_predictor():
    return Predictor(model=model)