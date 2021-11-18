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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.multioutput import MultiOutputClassifier

# class used to transform data to an acceptable format for GaussianNB
class DenseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///categorized_messages.db')
    df = pd.read_sql('select * from Cat',engine)[:100]
    
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
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Loaded data. \n')
        
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