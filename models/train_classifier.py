import sys
import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    """
    Load Data from the Database
    
    Arguments:
        database_filepath -> SQLite database
    Output:
        X -> Features dataframe
        Y -> Labels dataframe
        category_names -> name of categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_messages", con=engine)
    
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X,Y,category_names

def tokenize(text):
    """
    Tokenize, Lemmatizer, and remove stop words
    
    Arguments:
        text messages
    Output:
        clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop = stopwords.words("english")
    
    words = [tok for tok in tokens if tok not in stopwords.words("english")]
    
    clean_tokens = []
    for word in words:
        clean_tok = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build pipeline using random forest classifier
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print model accuracy with classification report
    
    """
    Y_pred = model.predict(X_test)
    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i], target_names=category_names))


def save_model(model, model_filepath):
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