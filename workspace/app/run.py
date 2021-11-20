import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
# import joblib
from sqlalchemy import create_engine

import pickle

# import functions from python module train_classifier
from classifier_module import tokenize, DenseTransformer


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
engine = create_engine('sqlite:///workspace/DisasterResponse.db')
df = pd.read_sql('select * from Cat', engine) # uncomment to run faster


# load model
with open("workspace/models/classifier.pkl", 'rb') as pickle_file:
    model = pickle.load(pickle_file)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
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
    # count of classes
    class_counts = df[Y_cols].sum().sort_values(ascending = False)[:8]
    
    # % of aid related
    aid_related = df['aid_related'].sum()
    totals = df[Y_cols].sum().sum()
    aid_related_rate = aid_related / totals *100
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=class_counts.index,
                    y=class_counts.values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classes, top 8 classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Class"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=['Aid Related Issues', 'Non aid related issues'],
                    values=[aid_related_rate, 100-aid_related_rate]
                )
            ],

            'layout': {
                'title': 'Fraction of aid related issues'
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