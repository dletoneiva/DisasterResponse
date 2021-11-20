# Disaster Response Pipeline Project
Final project for the data engineering course.

<li><a href='https://github.com/dletoneiva/DisasterResponse'>Repository for the files.</a></li>

## Motivation
Many disasters can be predicted. The final project for the Data Engineering course in Udacity's Data Scientist Nanodegree program is to develop a classificator that will get twitter messages as inputs and classify them according to their types, so specific responses can be taken more efficiently.

## Files in repository
_DisasterResponse.db_: SQLite database contained cleaned data.

/data:
_disaster\_categories.csv_: categorized training data.
_disaster\messages.csv_: messages training data.
_process\_data.py_: Python script to clean data and save it to an SQLite file.

/models:
_classifier.pkl_: Serialized ("pickled") file containing the model.
_train\_classifier.py: Python script to train the model and generate the pickle.

/app:
_classifier\_module.py_: Script available to import data for the pickle file.
_run.py_: Script to run the flask web app (use local IP+Port to run).
/app/templates:
_go.html_: Parser for model predictions.
_master.html_: Object that renders the model predictions.

## Quick start
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to your host IP adress with the port specified.

## Creator

Copyright Daniel Leto (dletoneiva@gmail.com). Please follow me on Twitter and Medium (@dletoneiva).
