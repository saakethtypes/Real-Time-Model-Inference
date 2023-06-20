# Real-Time-Model-Inference

This project is focused on training a Linear Regression model to predict future Return On Investment (ROI) for various advertising spend budgets across search, video, social media, and email channels. 

Tech stack includes - 
* Streamlit
* Snowpark
* Plotly
* Tensorflow
  
## Advertising Spend and ROI Prediction

* ML LR model training code on Snowflake using Python Stored Procedure
* Scalar and Vectorized Python User-Defined Functions (UDFs) for inference
* Snowflake Task to automate (re)training of the model
* Streamlit web application that uses the Scalar UDF for real-time inference on new data points based on user input
* Cohort Analysis on sales data for month on month sales w.r.t. unique users

![Output](https://github.com/saakethtypes/Real-Time-Model-Inference/assets/47172497/150ab681-ba89-4025-ba4b-2ecd85625c30)
