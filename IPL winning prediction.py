#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 


# In[2]:


match = pd.read_csv("C:\\Users\\Admin\\Downloads\\iplDatasets\\matches.csv")
delivery = pd.read_csv("C:\\Users\\Admin\\Downloads\\iplDatasets\\deliveries.csv")


# In[3]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[4]:


match.head()
delivery.head()


# # Total Runs Calculation

# In[5]:


total_score_df = total_score_df[total_score_df['inning'] == 1]


# In[6]:


total_score_df


# # Data Merging and Cleaning

# In[7]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
match_df


# # Team Cleaning and Filtering

# In[8]:


match_df['team1'].unique()


# In[9]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[10]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[11]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# # Filtering Matches and Features Creation

# In[12]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[13]:


match_df = match_df[['match_id','city','winner','total_runs']]


# In[14]:


delivery_df = match_df.merge(delivery,on='match_id')


# In[15]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[16]:


delivery_df


# # Feature Engineering

# In[17]:


delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce')
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()


# In[18]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[19]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[20]:


delivery_df


# # Wickets Calculation

# In[21]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')

# Convert the 'player_dismissed' column to numeric before using cumsum
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets'] = 10 - wickets
delivery_df.head()


# # 

# In[22]:


delivery_df.head()


# # Run Rate and Required Run Rate Calculation

# In[23]:


# crr = runs/overs
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[24]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# # Result Labeling

# In[25]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[26]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# # Final Data for Modeling 

# In[27]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[28]:


final_df = final_df.sample(final_df.shape[0])


# In[29]:


final_df.sample()


# In[30]:


final_df.dropna(inplace=True)


# In[31]:


final_df = final_df[final_df['balls_left'] != 0]


# # Data Splitting for Training and Testing

# In[32]:


X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X_train


# # Feature Transformation with One-Hot Encoding

# In[33]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')


# # Model Training - Logistic Regression, Random Forest, Support Vector Machine

# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# In[35]:


# Logistic Regression
logistic_pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

# Random Forest Classifier
rf_pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', RandomForestClassifier())
])

# Support Vector Machine
svm_pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', SVC())
])


# In[36]:


# Fit the models
logistic_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)
svm_pipe.fit(X_train, y_train)


# # Model Evaluation

# In[37]:


# Make predictions using the individual models
y_pred_logistic = logistic_pipe.predict(X_test)
y_pred_rf = rf_pipe.predict(X_test)
y_pred_svm = svm_pipe.predict(X_test)


# In[39]:


# Evaluate the models
print("Logistic Regression Accuracy:", logistic_pipe.score(X_test, y_test))
print("Random Forest Accuracy:", rf_pipe.score(X_test, y_test))
print("Support Vector Machine Accuracy:", svm_pipe.score(X_test, y_test))


# # Additional Metrics

# In[40]:


# Evaluate the models
from sklearn.metrics import accuracy_score, classification_report


# In[41]:


print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Support Vector Machine Accuracy:", accuracy_score(y_test, y_pred_svm))


# In[42]:


# Additional metrics (precision, recall, F1-score)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nSupport Vector Machine Classification Report:")
print(classification_report(y_test, y_pred_svm))


# # Match Summary Function

# In[43]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))


# # Match Progression Function

# In[44]:


def match_progression(x_df, match_id, model):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = model.predict_proba(temp_df)  # Use predict_proba for Random Forest
    temp_df['lose'] = np.round(result.T[0]*100, 1)
    temp_df['win'] = np.round(result.T[1]*100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-", target)
    temp_df = temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']]
    return temp_df, target


# In[45]:


temp_df, target = match_progression(delivery_df, 74, rf_pipe)
temp_df


# # Visualization of Match Progression

# In[46]:


import matplotlib.pyplot as plt

# Plot the data
plt.figure(figsize=(15, 5))
plt.plot(temp_df['end_of_over'].values, temp_df['wickets_in_over'].values, color='yellow', linewidth=3)
plt.plot(temp_df['end_of_over'].values, temp_df['win'].values, color='green', linewidth=4)
plt.plot(temp_df['end_of_over'].values, temp_df['lose'].values, color='red', linewidth=4)
plt.bar(temp_df['end_of_over'].values, temp_df['runs_after_over'].values)
plt.title('Target-' + str(target))
plt.show()


# # Prediction on New Data

# In[47]:


teams

delivery_df['city'].unique()


# In[48]:


import pandas as pd

columns = ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']

# Create a dictionary with example values for each feature
new_data_dict = {
    'batting_team': ['Sunrisers Hyderabad'],
    'bowling_team': ['Mumbai Indians'],
    'city': ['Mumbai'],
    'runs_left': [30],
    'balls_left': [36],
    'wickets': [3],
    'total_runs_x': [180],
    'crr': [8.0],
    'rrr': [8.5]
}

# Create a DataFrame from the dictionary
new_data = pd.DataFrame(new_data_dict, columns=columns)

# Now, 'new_data' represents an example of new data for prediction
print("New Data:")
print(new_data)


# In[49]:


predictions = rf_pipe.predict(new_data)

# Print the predictions 
print("Winning Prediction:")
print(predictions)


# In[50]:


import pandas as pd

columns = ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']

# Create a dictionary with modified example values for each feature
new_data_dict = {
    'batting_team': ['Royal Challengers Bangalore'],
    'bowling_team': ['Mumbai Indians'],
    'city': ['Bangalore'],
    'runs_left': [10],  # Increase the runs_left value
    'balls_left': [30],
    'wickets': [5],     # Decrease the number of wickets
    'total_runs_x': [190],
    'crr': [8.0],
    'rrr': [8.5]
}

# Create a DataFrame from the modified dictionary
new_data1 = pd.DataFrame(new_data_dict, columns=columns)

# Now, 'new_data' represents a modified example of new data for prediction
print("Modified New Data:")
print(new_data1)


# In[51]:


predictions = rf_pipe.predict(new_data1)

# Print the predictions 
print("Winning Prediction:")
print(predictions)
 


# In[ ]:




