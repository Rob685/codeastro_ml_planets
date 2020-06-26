# Exo_Predict

This machine learning planet classification package allows users to predict the number of exoplanets a user is likely to find in a given exoplanet system. It does this based solely on the stellar mass, radius, temperature, and the exoplanet discovery method employed.

## Functionality

Using the default `exo_predict` module, you can use our model trained on all confirmed planets from the NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/) in order to predict whether or not the systems in your data contain one or multiple planets.   

If you'd rather use your own data train a model, you can feel free to do so with our `exo_model_predict module`. It takes in your training data to create a model, then hands you back the model and a testing dataset. This module also allows you to input new data to try out once you've tried out your model with the `predict_exoplanets` function.

## Tutorials

Two tutorials are included with some sample data so you can try this out for yourself! Find them in the Tutorials directory.  


## Requirements

You will need the following to use this package:

numpy
pandas
sklearn
xgboost
matplotlib
pickle



