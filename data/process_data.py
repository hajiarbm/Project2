import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    '''
     load_data
     load data from files and merge them to create a dataframe
     
     Input:
     messages_filepath     file path to messages csv file
     categories_filepath   file path to categories csv file 

     Returns:
     df    Merged messages and categories information
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df

def clean_data(df):
    '''
    clean_data
    cleans the data, remove nans and create new columns

    Input:
    df    Dataframe based on messages and categories

    Returns:
    df    The cleaned dataframe
    '''
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [row[i][:-2]for i in range(len(row))]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = [categories[column][i][-1] for i in range(len(categories))]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(columns=['categories'], axis=1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis =1)
    # drop duplicates
    df = df.drop_duplicates()
    return df
    
    
def save_data(df, database_filename):
   '''
   save_data
   saves the data in the database

   Input:
   df    Dataframe based on messages and categories
   '''
   path = 'sqlite:///'+database_filename  #+'.db'
   print(path)
   engine = create_engine(path)
   df.to_sql('mytable', engine, index=False,if_exists='replace' )  #####
   print(database_filename)

def main():
    '''
    main
    main method to call the methods
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
