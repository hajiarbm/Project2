# Project2
Disaster Response project tries to improve the efficiency in critical moments. It is based on messages that were sent during disasters.

The benifits of this project can be:

1) Determine the resources and in case of shortage, find the priority.

2) By analyzing the messages it can identify the needs.

3) Understanding the text can be helpful to design a system that can understand messages in defferent languages.

4) Can be used to categorize the emergencies effectively.
************************************************************************************

Process_data.py contains the ETL pipeline that cleans data and stores in database.

train_classifier.py runs ML pipeline that trains classifier and saves the information.

run.py creates plots on the web


- Root Directory
    - data
        - process_data.py
        - disaster_messages.csv
        - disaster_categories.csv
  
    - models
        - train_classifier.py
    - app
        - run.py
        - templates
            - master.html
            - go.html
      

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

