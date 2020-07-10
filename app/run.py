import json
import plotly
import pandas as pd
import sys

sys.path.append("/home/workspace/models")

from custom_transformer import CustomTransformer
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.figure_factory as ff

from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3

app = Flask(__name__)


# load data
con = sqlite3.connect('data/DisasterResponse.db')
df = pd.read_sql_query("SELECT * from messages", con)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_sum = df.groupby('genre').sum()
    genre_names = list(genre_sum.index)
    labels = list(genre_sum.columns)[1:]
    social_freq = df[df.genre == 'social'][[col for col in df.columns if col not in ['message','original','genre','id']]].sum(axis=1)
    direct_freq = df[df.genre == 'direct'][[col for col in df.columns if col not in ['message','original','genre','id']]].sum(axis=1)
    news_freq = df[df.genre == 'news'][[col for col in df.columns if col not in ['message','original','genre','id']]].sum(axis=1)
    # create visuals
    
    hist_data = [direct_freq, news_freq, social_freq]
    group_labels = ['Direct','News','Social']
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
    #fig.update_layout(title_text='Distribution of Number of Labels per Message')
    fig.update(layout=dict(title=dict(text='Distribution of Number of Labels per Message', x=0.5)))
    freq_graph = fig.to_dict()
    
    graphs = [
        
        {
            'data': [
             {'values': genre_sum.loc['direct'][1:],
             'labels': labels,
             'domain': {'x':[0,.32]},
             'name': 'Direct',
             'title':'Direct',
             'hoverinfo': 'label + percent + name',
             'textposition':'inside', 
             'hole': .3,
             'type': 'pie',
             'scalegroup': 'one'},
            {'values': genre_sum.loc['news'][1:],
             'labels': labels,
             'domain': {'x':[0.34,.65]},
             'name': 'News',
             'title':'News',
             'hoverinfo': 'label + percent + name',
             'textposition':'inside', 
             'hole': .3,
             'type': 'pie',
             'scalegroup': 'one'},
             {'values': genre_sum.loc['social'][1:],
             'labels': labels,
             'domain': {'x':[0.67,1]},
             'name': 'Social',
             'title':'Social',
             'hoverinfo': 'label + percent + name',
             'textposition':'inside', 
             'hole': .3,
             'type': 'pie',
             'scalegroup': 'one'}
            ],

            'layout': {
                'height': 550,
                'width': 1200,
                'legend': dict(y=-.01,orientation='h'),
                'title': dict(text = 'Distribution of Message Labels by Genres', 
                              yref = "paper",y = 1.1, yanchor = "bottom" )
      
            }
        },
        
        {
         'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Total Number of Messages by Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        freq_graph
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
