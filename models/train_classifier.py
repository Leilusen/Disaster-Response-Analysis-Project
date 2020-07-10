import sys
import sqlite3
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
wpt = nltk.WordPunctTokenizer()

from custom_transformer import CustomTransformer
from custom_transformer import tokenize

def display_results(y_test, y_pred, categories):
    for i, cat in enumerate(categories):
        print('{}: \n'.format(cat))
        #print('{}: \n'.format(df[[col for col in df.columns[4:]]].columns[i]))
        print(classification_report(y_test[:,i],y_pred[:,i]))
        
def load_data(database_filepath):
    con = sqlite3.connect(database_filepath)
    df = pd.read_sql_query("SELECT * from messages", con)
    X = df['message'].values
    Y = df[[col for col in df.columns[4:]]].values
    categories = [col for col in df.columns[4:]]

    return X, Y, categories

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(ngram_range = (1, 6), max_df = 0.65,
                                         max_features = 12500, tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('custom_trans', CustomTransformer())
        ])),

        ('clf', MultiOutputClassifier(SGDClassifier(n_jobs=-1, random_state = 42, alpha = 0.0001,
                                                   penalty='l2', class_weight = None,
                                                   loss = 'hinge', shuffle = False, average = False,
                                                   )))
    ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1,6), (1,7), (1,8)),
        #'features__text_pipeline__vect__max_df': (0.65, 0.68, .71),
        #'features__text_pipeline__vect__max_features': (12000, 12500, 13000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__alpha': [0.00001, 0.00005, 0.0001, 0.0005], 
        #'clf__estimator__average': [True, False], 
        #'clf__estimator__class_weight': [None, 'balanced'], 
        #'clf__estimator__early_stopping': [True, False], 
        #'clf__estimator__epsilon': 0.1, 
        #'clf__estimator__eta0': [0.001, 0.01, 0.1, 0.5], 
        #'clf__estimator__learning_rate': ['optimal','invscaling'], 
        #'clf__estimator__loss': ['hinge','log', 'modified_huber','squared_hinge'], 
        #'clf__estimator__max_iter': 1000, 
        #'clf__estimator__n_iter_no_change': 5, 
        #'clf__estimator__penalty': ['l1', 'l2', 'elasticnet'], 
        #'clf__estimator__shuffle': [True,False], 
        #'features__transformer_weights': (
            #{'text_pipeline': 1, 'custom_trans': 0.5},
            #{'text_pipeline': 0.5, 'custom_trans': 1},
            #{'text_pipeline': 0.8, 'custom_trans': 1})

    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, categories):
    Y_pred = model.predict(X_test)
    display_results(Y_test, Y_pred, categories)
    
    pass


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

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