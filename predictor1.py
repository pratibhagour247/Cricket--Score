import pandas as pd
import os
import ast
import json
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#getting label encoders and then manually creating encoder json (to omit the repetetive grounds)
def create_encoder():
    le = preprocessing.LabelEncoder()
    dataset = pd.read_csv('to_be_trained.csv')
    #using this file for creating json
    label_encoder = open("label_encoder.txt",'w')
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            dataset[column] = le.fit_transform(dataset[column])
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            label_encoder.write(str(mapping))
    label_encoder.close()

def transformDataset():
    dataset = pd.read_csv('to_be_trained.csv')
    dataset = dataset.groupby(['match_id', 'innings',], as_index=False).apply(lambda x: x[x.batsman_count == x.batsman_count.max()])
    dataset = dataset.groupby(['match_id', 'innings',], as_index=False).apply(lambda x: x[x.bowler_count == x.bowler_count.max()])
    dataset['total_runs_cumulative'] = dataset['total_runs'].max()

    #group by max of bowler name count
    dataset = dataset.groupby(['match_id', 'innings',], as_index=False).apply(lambda x: x[x.total_runs == x.total_runs.max()])
    #final grouping for training model
    dataset = dataset.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team','batsman_count','bowler_count','total_runs'], as_index=False).agg(total_runs_cumulative = ('total_runs_cumulative','mean')).reset_index()
    dataset = dataset.drop('total_runs_cumulative', 1)
    dataset = dataset.drop('index', 1)
    dataset = dataset.drop('match_id', 1)
    dataset.to_csv(r'to_be_trained.csv', index=False)


    #creating encoder.json (this will be done manually to remove repetition)
    create_encoder()


def formatter(file):
    df = pd.read_csv(file)
    match_id = df['match_id'].tolist()
    innings = df['innings'].tolist()
    old_innings = innings[0]
    old_match =  match_id[0]
    new_batsmen_list = set()
    new_bowler_list = set()
    total_runs_list = []
    for idx, row in df.iterrows():
        # clubbing up as per match id and innings
        if match_id[idx] == old_match and innings[idx] == old_innings:
            pass
        else:
            new_batsmen_list = set()
            new_bowler_list = set()
            total_runs_list = []
            total_runs = 0
        new_batsmen_list.add(df.at[idx, 'striker'])
        new_batsmen_list.add(df.at[idx, 'non_striker'])
        new_bowler_list.add(df.at[idx, 'bowler'])
        total_runs_list.append(df.at[idx, 'total_runs'])
        total_runs = sum(total_runs_list)
        #total batman played and bowler bowled in 6 over
        #(as we cant match batsman name and bowler name from input sheet as they have different names)
        df.at[idx, 'batsman_count'] = len(new_batsmen_list)
        df.at[idx, 'bowler_count'] = len(new_bowler_list)
        df.at[idx, 'total_runs'] = total_runs        
        old_innings = innings[idx
        ]
        old_match = match_id[idx]

    #we dont need this anymore
    df = df.drop('striker', 1)
    df = df.drop('non_striker', 1)
    df = df.drop('bowler', 1)


    df.to_csv(r'to_be_trained.csv', index=False)

    #doing some more final transformation
    transformDataset()


def transformInitialDataset(): 

    # Importing the initial all matches dataset
    dataset = pd.read_csv('ipl_csv2/all_matches.csv')  # ye main file 
    # transforming dataset such that we have 2 innings and overall 6 over , also  total runs in 6 over
    dataset = dataset[dataset['ball'] < 6.0]    
    dataset = dataset[dataset['innings'] < 3]  
    dataset['total_runs'] = dataset['runs_off_bat'] + dataset['extras']
    dataset = dataset.groupby(['match_id', 'venue', 'innings', 'batting_team','striker','non_striker','bowler', 'bowling_team','total_runs'], as_index=False).agg(total_runs = ('total_runs','sum')).reset_index()
    print(dataset)
    dataset.to_csv(r'to_be_trained.csv', index=False)

    #calling second tranformation method
    # formatter('to_be_trained.csv')

    
# def algo(venue,innings,batting_team,bowling_team,batsmen,bowlers):

#     le = preprocessing.LabelEncoder()
#     # Importing the training dataset
#     dataset = pd.read_csv('to_be_trained.csv')
#     for column in dataset.columns:
#         if dataset[column].dtype == type(object):
#             dataset[column] = le.fit_transform(dataset[column])
#     X = dataset.iloc[:,[0,1,2,3,4,5]].values
#     y = dataset.iloc[:, 6].values

    # Plot outputs
    # sns.catplot(x="innings", y="total_runs",kind="bar", data=dataset)
    # #sns.catplot(x="batsman_count", y="total_runs", data=dataset)
    # #sns.catplot(x="batting_team", y="total_runs", data=dataset)
    # #sns.relplot(x="batsman_count", y="innings", hue="total_runs",sizes=(15, 200), data=dataset)
    #sns.regplot(x="bowler_count", y="total_runs", data=dataset)
    #print(plt.show())


    # Splitting the dataset into the Training set and Test set
    # seed = 42
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = seed)
    # pipe = make_pipeline(StandardScaler(), LogisticRegression())


    # Training the dataset

    # pipe.fit(X_train,y_train)
    # # checking score
    # #print(pipe.score(X_test,y_test)*100)
    
    # # save the model to disk
    # save_model = 'linear_model.aditya'
    # joblib.dump(pipe, save_model)

    # # Testing with a custom input

    # loaded_model = joblib.load('linear_model.aditya')

    # new_prediction = loaded_model.predict(np.array([[venue,innings,batting_team,bowling_team,batsmen,bowlers]]))
    # print("Prediction score:" , int(round(new_prediction[0])))



#     return int(round(new_prediction[0]))


# def final_predict(venue,innings,batting_team,bowling_team,batsmen,bowlers):

#     loaded_model = joblib.load('logistic_model.aditya')

#     new_prediction = loaded_model.predict(np.array([[venue,innings,batting_team,bowling_team,batsmen,bowlers]]))
#     print("Prediction score:" , int(round(new_prediction[0])))

#     loaded_model = joblib.load('linear_model.aditya')

    # new_prediction = loaded_model.predict(np.array([[venue,innings,batting_team,bowling_team,batsmen,bowlers]]))
    # print("Prediction score linear:" , int(round(new_prediction[0])))

    # return int(round(new_prediction[0]))

def predict():
    #importing and transform the initial all matches file
    #uncomment this when you are training the model
    transformInitialDataset()

    #getting input file
    input_file = pd.read_csv('inputFile.csv')
    # get integer values of these from encoder.json

    with open("encoder.json") as f:
        new_dict = json.load(f)
    for idx, row in input_file.iterrows():
        new_venue = str(input_file.at[idx, 'venue']).strip()
        if new_venue.find('"') != -1:
            new_venue = new_venue.replace('"',"'")
        venue = new_dict['venue'][new_venue]
        innings = input_file.at[idx, 'innings']
        batting_team = new_dict['teams'][str(input_file.at[idx, 'batting_team']).strip()]
        bowling_team = new_dict['teams'][str(input_file.at[idx, 'bowling_team']).strip()]
        batsmen = input_file.at[idx, 'batsmen'].count(",") + 1
        bowlers = input_file.at[idx, 'bowlers'].count(",") + 1

    #training model 
    #return algo(venue,innings,batting_team,bowling_team,batsmen,bowlers)
        
    #firing final prediction method
    # return final_predict(venue,innings,batting_team,bowling_team,batsmen,bowlers)
        

def predictRuns():
    prediction = 0
    prediction = predict()
    return prediction

predictRuns()