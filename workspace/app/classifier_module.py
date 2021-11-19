import sys

# natural language toolkit
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# data and numeric libraries
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# model libraries
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier

# model storage
import pickle 

# 
class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Class used to transform data to an acceptable format for GaussianNB

    Args:
        BaseEstimator ([object]): sklearn estimator
        TransformerMixin ([object]): sklearn transformer
    
    Returns:
        Transformed model
    """    

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def load_data(database_filepath):
    """
    Loads data to be used in the model

    Args:
        database_filepath ([path]): Path for the SQL database.

    Returns:
        [type]: [description]
    """    
    # load data from database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('select * from Cat',engine) [:100] # uncomment to run faster


    
    # define columns to be used
    Y_cols = ['related', 'request', 'offer',
          'aid_related', 'medical_help', 'medical_products', 
          'search_and_rescue', 'security', 'military', 
          'child_alone', 'water', 'food', 'shelter',
          'clothing', 'money', 'missing_people', 'refugees', 
          'death', 'other_aid', 'infrastructure_related', 
          'transport', 'buildings', 'electricity', 'tools', 
          'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
          'weather_related', 'floods', 'storm', 'fire', 
          'earthquake', 'cold', 'other_weather', 'direct_report']
    
    # load values into vars
    X = df['message']
    Y = df[Y_cols]

    return X, Y, Y_cols

def tokenize(text):
    """
    Tokenizes text data, lemmatizes it and removes URL's.

    Args:
        text ([string]): String containing the text to be tokenized.

    Returns:
        [list of strings]: List containing tokenized text
    """    
    # filter URL's using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize and lematize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """
    Instantiates the model to be trained.

    Returns:
        [object]: sklearn object.
    """    
    # build it as a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('toarray', DenseTransformer()),
        ('clf', MultiOutputClassifier(BernoulliNB())) # great for binary classification
    ])

    # use gridsearch for hyperparameters tuning
    parameters = {
        'clf__estimator__alpha': (0.1, 0.5, 0.9)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model, printing all the scores using classification_report.

    Args:
        model ([object]): Model to be evaluated.
        X_test ([object]): Pandas dataframe object containing the text (1D).
        Y_test ([object]): Pandas dataframe object containing the classes.
        category_names ([list of strings]): Category names to be evaluated.
    """    
    # predict based on X_test
    y_pred = model.predict(X_test)

    # transform to dataframe to be easier to read
    df_y_pred = pd.DataFrame(y_pred)
    df_y_pred.columns = category_names

    for col in category_names:
        print('Analysed class: '+ str(col) + '\n')
        print('Bernoulli Naive Bayes (with GridSearch) classification: \n')
        print(classification_report(Y_test[col], 
                                    df_y_pred[col]))



def save_model(model, model_filepath):
    """
    Saves the model in selected path.

    Args:
        model ([object]): Trained model to be saved.
        model_filepath ([path]): Filepath for pickle file.
    """    
    pickle.dump(model, open(model_filepath,'wb'))