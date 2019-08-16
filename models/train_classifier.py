 ## import libraries
import sys
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])



def load_data(database_filepath):
    '''
    input:
        database file
    output:
        X and Y, Messages and Categories and name of categories
        
       '''
    engine = create_engine('sqlite:///'+'DisasterResponse.db')
    df = pd.read_sql_table('disaster' , engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_name = list(df.columns[4:])
    
    return X , Y, category_name


def tokenize(text):
    '''
    input:
    clean text data
    
    output:
    tokenized and lemmatized text
   '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    ## normalize text
    detected_urls = re.findall(url_regex , text)
    
    for url in detected_urls:
        text = text.replace(url , "urlplaceholder")
        
    tokens = word_tokenize(text) # tokenize text
    lemmatizer = WordNetLemmatizer()
    
    words = [word for word in tokens if word not in stopwords.words('english')]
    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

    
    

def build_model():
    '''
     a ML pipeline using random forest classifier with gridsearch 
     '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf' , MultiOutputClassifier(RandomForestClassifier()))
                        ])
        
    parameters = {'clf__estimator__n_estimators':[50],
                  'clf__estimator__min_samples_split':[5]
                  ## 'clf__estimator__criterion':['gini','entropy'],
                  ## 'clf__estimator__max_features':['auto','log2']
                 }
        
    cv = GridSearchCV(pipeline, param_grid = parameters , n_jobs = 4 ,cv=3)  
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    test trained model on test data 
     output:
     accuracy, recall, and precision for all 36 categories
     
     '''
    pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, pred[:,i])))
    


def save_model(model, model_filepath):
    ''' 
    export model as a pickle file
    
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()