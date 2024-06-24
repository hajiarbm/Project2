# import libraries
import sys
import pandas as pd
import sqlite3
from sqlalchemy import *
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle


def load_data(database_filepath):
    '''
    load_data
    loads the dataframe

    Input:
    database_filepath    The path to the database
    
    Returns:
    X    The data
    Y    The targets
    categories The columns of the dataframe
    '''
    
    # load data from database
    path = 'sqlite:///'+database_filepath
    
    engine = create_engine(path)
    df = pd.read_sql_table('mytable', engine)  
    X = df.drop([ 'id', 'original' , 'genre' , 'related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report'], axis=1)
    Y = df[[ 'related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    categories = [ 'related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    return X, Y, categories

def tokenize(text):
    '''
    tokenize
    Gets a text and extract the important words from it

    Input:
    text    A text message

    Returns:
    clean_tokens    important words from the message    
    '''
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    Create the pipeline

    Returns
    pipeline    Returns the pipeline
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    evaluate the model on the test data

    Input:
    model    The pipeline
    X_test   The test data
    Y_test   Target of the test data
    category_names    the names of the categories
    '''
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(Y_test)
    y_pred = pd.DataFrame(y_pred)
    # Evaluate the model on the test data
    for i in range(y_test.shape[1]):
        print(f"Classification Report for Output {i + 1}:")
        print(classification_report(y_test.iloc[:, i], y_pred.iloc[:, i]))

def save_model(model, model_filepath):
    '''
    save_model
    
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X['message'], Y, test_size=0.2)
        
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
