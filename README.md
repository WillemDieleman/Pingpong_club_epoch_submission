This submission folder contains the files used by the team Pingpong club Epoch for the Axpo x Databricks x ETH Analytics Club Iberia Retail Consumption Forecasting challenge. 

The structure of this submission is the following:

- **final_submission**: This notebook contains the inference code for our final submission. 
- **AutogluonModels/**: The model for the final submission is stored here. Its ML-flow name is dark-ember-57. 
- **src/**: This folder contains all the note books we used during the datathon. Our final submission does not use any functions from here. The code and files is not cleaned up. 

The pipeline for our submission is the following:

- Aggregating all the data per region into a single number per timestamp (18 regions)
- Adding open source weatherdata for a bunch of spanish cities and averinging them to get a forecast for weather features like temperature, humidity, and rainfall per region. 
- Adding national and regional holidays to the featureset using the holidays python package. 
- Loading the autogluon-trained ensemble of ChronosplusRegressor, GBDT ensemble and LSTM.
- Predicting December 1st based on all the data up until november, weather forecase and holiday information for each of the 18 regions and summing the outputs.
- Adding the actual powerusage of December 1st to all the data to predict December 2nd, and repeat until Febuary 2026.

An Databricks App of a dashboard comparing the performance of our models compared to the baseline is also available at https://forecast-dashboard-7474648427238243.aws.databricksapps.com/. 

