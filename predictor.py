### Custom definitions and classes if any ###

import pandas as pd

def importDataSet():
    # Importing the dataset
    dataset = pd.read_csv('../ipl_csv2/1254060.csv')
    x = dataset.iloc[:,[3,4,5,6,7,8,9]].values
    y = dataset.iloc[:, 14].values
    print(x)

def predictRuns(testInput):
    prediction = 0
    importDataSet()
    return prediction

predictRuns(0)