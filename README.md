# Distaster Response Pipeline

## Summary

Using messages received during different kind of disasters, this project delivers a web application capable of classifying incoming disaster related messages into different categories. This enables proper routing of messages to the correct disaster relief agency.

A machine learning model in the backend provides this functionality. The project contains a trained model, scripts for training such a model yourself and a data pipeline for processing text data. The final product is a web application which can classify incoming messages.

## Usage

Install the required packages for the project using pip:

`pip install -r requirements.txt`

To run ETL pipeline that cleans data and stores in 
database:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

To run ML pipeline that trains classifier and saves it:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Run the following command in the app's directory to run the web app:

`python run.py`

Go to http://0.0.0.0:3001/

## Files

The project is divided into three folders, one to contain the data, another for the model and another for the web application.

├── app  
│   ├── run.py  
│   └── templates  
│       ├── go.html  
│       └── master.html  
├── data  
│   ├── DisasterResponse.db  
│   ├── disaster_categories.csv  
│   ├── disaster_messages.csv  
│   └── process_data.py  
├── models  
│   ├── README.md  
│   ├── classifier.pkl  
│   ├── train_classifier.py  
│   └── workspace_utils.py  
├── requirements.txt
