import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from selected file (both messages and categories).

    Args:
        messages_filepath ([path]): Path for csv file containing messages.
        categories_filepath ([path]): Path for csv file containing categories.

    Returns:
        [object]: Pandas dataframe with both merged.
    """    

    # load messages dataset
    messages = pd.read_csv('data/disaster_messages.csv')

    # load categories dataset
    categories = pd.read_csv('data/disaster_categories.csv')
    idx = categories['id']

    # merge datasets
    df = pd.merge(messages,
                categories,
                on='id',
                how='inner')
    
    return df

def clean_data(df):
    """
    Cleans data obtained through load_data().

    Args:
        df ([object]): Pandas dataframe to be cleaned.

    Returns:
        [object]: Cleaned dataframe.
    """    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # extract list of categories using the first row
    category_colnames = categories.iloc[0,:].str.split('-', expand=True)[0].values

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1].values
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # add indexes
    categories['id'] = df['id']

    # drop the original categories column from `df`
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories,
            on='id',
            how='inner')

    # drop duplicates
    df = df.drop_duplicates()

    # remove rows that are not binary
    df = df.drop(df.loc[~df[column].isin([0,1])].index)['related'].unique()

    return df

def save_data(df, database_filename):
    """
    Saves cleaned data to a SQL database.

    Args:
        df ([object]): Pandas database to be saved.
        database_filename ([string]): Name of file to be saved.
    """    
    # save everything to a database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('Cat', engine, index=False)  

def main():
    print(sys.argv[1:])
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