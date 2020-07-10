# Disaster response analysis
Web application designed to analyze and label messages received by the disaster response hub.

There are 36 categories in total, however due to absence of any messages related to "Child alone" category in the training dataset, we removed this category from our analysis.

The dataset is heavily imbalanced. SMOTE for multioutput classification could be used to further improve results.
### Files in the repository:
1. "data" folder:
 - "disaster_messages.csv" - file containing all messages and their sources ("genre" field). Sources include direct messages, social media and news.
 - "disaster_categories.csv" - file containing messages labeled by 36 categories depending on the actions involved.
 - "process_data.py" - script that performs data cleaning and uploads final dataframe into SQLite database.
 
2. "models" folder:
 - "train_classifier.py" - script containing ML pipeline that performs grid-search for SGD classifier and saves resulting model as pkl file in the directory.
 - "custom_transformer.py" - data transformer used for feature engineering

3. "app" folder:
 - "templates" folder contains HTML templates for the web-app
 - "run.py" - file with the Flask app. It uploads data from SQLite database, demonstrates some visualizations related to the training dataset, performs analysis and categorization of any user defined message.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
