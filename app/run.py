import sys
import json
import plotly
import pandas as pd
import numpy as np 

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify

from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine
from collections import Counter
from sklearn.externals import joblib


app = Flask(__name__)

def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterMessages.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract top 10 categories
    #  counts
    top_values = df.iloc[:,4:].sum().sort_values(ascending=False)[1:11]
    # labels
    top_labels = list(top_values.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data':[
                Bar(
                    x= top_labels,
                    y= top_values
                )
            ],
            
            'layout':{
                'title' : 'Top Ten Labels',
                'xaxis': {
                    'title' : "Labels"
                },
                'yaxis':{
                    'title': "Count"
                }
            }
        }         
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
     app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()