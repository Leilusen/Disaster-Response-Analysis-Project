import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Function for loading data from specified directories
    Parameters:
    Messages and categories files' directories
    Returns:
    Combined dataframe
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_index=True,right_index=True)
    df.drop('id_y', axis=1, inplace = True)
    df.rename(columns={'id_x': 'id'}, inplace = True)
    return df


def clean_data(df):
    '''
    Function for cleaning data and preparing it for analysis
    Parameters:
    Dataframe containing messages and their corresponding categories
    Returns:
    Cleaned dataframe
    
    '''
   # create a dataframe of the 36 individual category columns

    categories = df.categories.str.split(';', expand = True)

    categories.columns = [categories.loc[0,col].split('-')[0] for 
                          col in categories.columns]

    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        categories[column] = categories[column]
        
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # drop duplicated records
    df = df[(~df.duplicated()) & (df['related']!=2)]
    df.drop('child_alone', axis = 1, inplace = True)
    return df


def save_data(df, database_filename):
    '''
    Function for saving data into SQLite database
    Parameters:
    Dataframe and database directory
    Returns:
    None
    
    '''
    # loading to the database
    conn = sqlite3.connect(database_filename)
    
    # saving the data to the table
    df.to_sql('messages', conn, if_exists = 'replace', index=False) 
    
    return None


def main():
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
