# Project2
Disaster Response project tries to improve the efficiency in critical moments. 

The benifits of it can be

Determine the resources and in case of shortage, find the priority.

By analyzing the messages it can identify the needs.


************************************************************************************

Process_data.py contains the ETL pipeline that cleans data and stores in database.

train_classifier.py runs ML pipeline that trains classifier and saves the information.

run.py creates plots on the web

# Disaster Response Pipeline Project

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    - 
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
      
    - To run ML pipeline that trains classifier and saves
    - 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

