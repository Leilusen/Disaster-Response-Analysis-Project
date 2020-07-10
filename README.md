# Disaster response analysis
Web application designed to analyze and label messages received by the disaster response hub. 
There are 36 categories in total, however due to absence of any messages related to "Child alone" category in the training dataset, we removed this category from our analysis.
The dataset is heavily imbalanced. We used multioutput classifier in our pipeline. As a future improvement, some kind of SMOTE for multioutput could be used to improve classification.
### Files in the repository:
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
