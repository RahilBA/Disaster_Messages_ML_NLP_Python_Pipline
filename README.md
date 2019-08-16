
# Disaster Response Pipeline Project
This project is part of the Data Science Nonodegree Program by Udacity and data is provided by Figure Eight.

### Objective: 
The purpose of this project is to build a Natural Language Processing classifier that categorize the tweets and actual messages. 

### This project is divided to 3 parts:
1) Create ETL Pipeline to get and clean data, also save in a SQLlite database. 
2) Create a Machine Learning Pipeline to train a NLP classifier. 
3) Create an web App to show the result in Heroku. 

### Installation 
The following packages need to be installed:
stopwords
punkt
wordnet

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterMessages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

