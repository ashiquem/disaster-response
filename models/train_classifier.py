import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from workspace_utils import active_session
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    """ Loads database object from filepath and returns extraced text and labels """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterMessages',engine)
    engine.dispose()
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    return X,Y


def tokenize(text):
    """ Tokenizes input text and returns tokens. 
    
    Keyword arguments:
    text(str) -- input text
    
    Returns:
    tokens(list of str) -- list of tokens
    
    """

    #normalize
    text = text.lower()
    #tokenize
    words = word_tokenize(text)
    #remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    #lemmatize
    tokens = [WordNetLemmatizer().lemmatize(w.strip()) for w in words]
    
    return tokens

def build_model():

    """ Builds and returns scikitlearn GridCV object with pipeline.


    Returns:
    cv(GridSearchCV) -- grid search model object
    
    """
    
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
    
    grid = {'clf__estimator__n_estimators': [10, 20,30],
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__min_samples_split': [2, 5, 10]
       }
    
    cv = GridSearchCV(pipeline, param_grid=grid, n_jobs=-1, cv=2)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    
    """ Evaluates model and prints out classification results.

    Keyword arguments:
    model(sklearn pipeline) -- model object
    X_test(pandas df) -- messages
    Y_test(pandas df) -- labels
 
    """

    ypreds = model.best_estimator_.predict(X_test)
    
    for ind,col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col],ypreds[:,ind]))
    
    return

def save_model(model, model_filepath):
    
    """Saves model object to file."""

    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        with active_session():
            print('Training model...')
            model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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