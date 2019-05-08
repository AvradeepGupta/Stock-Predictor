import pandas as pd
import quandl , math , datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression

style.use('ggplot')                                                                 #specify the style of plotting

df = quandl.get('WIKI/GOOGL')                                                       #getting data

#print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]            #initial dataframe

df['HL_PCT']=(df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0        #new feature
df['PCT_change']=(df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0     #new feature

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]                         #final dataframe

#print(df.head())                                                                    #printing first five rows


forecast_col = 'Adj. Close'
df.fillna(-9999,inplace=True)                                                       #filling empty spots in data

forecast_out = int(math.ceil(0.01*len(df)))                                         #0.01 is used to shift the label data by a few days
#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)


#print(df.tail())                                                                   printing last five rows


X = np.array(df.drop(['label'],1))                                                  #dropping label column and everything else is a dataset
X = preprocessing.scale(X)                                                          #scaling our dataset
X = X[:-forecast_out]
X_lately = X[-forecast_out:]                                                        #X's for which we dont have a Y value


df.dropna(inplace=True)
#Y = np.array(df['label'])                                                          #our label is the label column
Y = np.array(df['label'])


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)              #.2 or 20% of the dataset is used to train


clf = LinearRegression(n_jobs=-1)                                                   #setting up the linear regression model njobs means number of parallel processes
clf.fit(X_train, Y_train)                                                           #training the model
accuracy = clf.score(X_test,Y_test)                                                 #checking accuracy
#print("Accuracy of the model is",accuracy*100,"%")                                 #printing accuracy


#Prediction

forecast_set = clf.predict(X_lately)
#print(forecast_set, accuracy , forecast_out)                                       #to see the projected values

#Printing

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400                                                                     #number of seconds in a day basically
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]            #loc refers to the index of the dataframe

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()









