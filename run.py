import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


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

#sqlite:///data/DisasterResponse.db
engine = create_engine('sqlite:///../data/DisasterResponse.db')

df = pd.read_sql_table('InsertTableName2', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # Extract data for second graph
    genre_aid_sum = df.groupby('genre')['aid_related'].sum()
    genre_names2 = list(genre_aid_sum.index)
    
    
    # Extract data for third graph
    genre_weather_sum = df.groupby('genre')['weather_related'].sum()
    genre_names3 = list(genre_weather_sum.index)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
            'data': [
                Bar(
                    x=genre_names2,
                    y=genre_aid_sum
                )
            ],
            'layout': {
                'title': 'Distribution of Aid Related Genres',
                'yaxis': {
                    'title': "Sum"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=genre_names3,
                    y=genre_weather_sum
                )
            ],
            'layout': {
                'title': 'Distribution of Weather Related Genres',
                'yaxis': {
                    'title': "Sum"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


    # Second plot
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_aid_sum = df.groupby('genre')['aid_related'].sum()
    #genre_names2 = list(genre_aid_sum.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    #graphs2 = [
     #   {
      #      'data': [
       #         Bar(
        #            x=genre_names2,
         #           y=genre_aid_sum
          #      )
           # ],

            #'layout': {
             #   'title': 'Distribution of Aid Related Genres',
              #  'yaxis': {
               #     'title': "Sum"
                #},
               # 'xaxis': {
                #    'title': "Genre"
               # }
           # }
       # }
   # ]
    
    # encode plotly graphs in JSON
    #ids = ["graph-{}".format(i) for i, _ in enumerate(graphs2)]
    #graphJSON2 = json.dumps(graphs2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    #return render_template('master.html', ids=ids, graphJSON=graphJSON2)


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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()