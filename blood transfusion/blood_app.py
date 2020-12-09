import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn import metrics



st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Blood Transfusion Predictor..')

DATA_URL = (r'C:\Users\Sam\Desktop\Notitia Project\Notitia-Data-Science\project\blood transfusion\transfusion.csv')

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)

    return data

#blood_t = load_data()

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

# Inspect Data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    st.text("Recency - months since the last donation\n Frequency - total number of donation\n Monetary - total blood donated in c.c. \n Time - months since the first donation \n a binary variable representing whether he or she donated blood in March 2007 1 stands \n for donating blood; 0 stands for not donating blood")

# Show data shape

if st.checkbox('Show Dataset Shape'):
    st.write('Shape of dataset:', data.shape)

data.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)

count = data['target'].value_counts()
sns.countplot(x='target', data=data)
#fig = plt.show()

if st.checkbox('Display Total NUmber of Donors and Non donors: [1 = Donors] [0 = Non Donors]'):
    st.write(count)
    st.pyplot()
    


# Declaring Input Variables and Target Variable
X = data.iloc[:, [0,1,3]].values
Y = data.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to Training set
classifier = LogisticRegression(random_state=0, solver='lbfgs')
classifier.fit(X_train, Y_train)

# Predicting Test set results to divide dependent variables into zeros and ones 
Y_pred = classifier.predict(X_test)

accuracy = metrics.accuracy_score(Y_test, Y_pred)

if st.checkbox('Show Prediction Accuracy'):
    st.write(accuracy)

# Measuring Classificaton performance with Confusion Matrix

cm = confusion_matrix(Y_test, Y_pred)

if st.checkbox('Confusion Matrix'):
    st.write(cm)
    st.text('129 == Obsereved Zeros\n 5   == Observed Ones\n 3   == Number of False Positves \n 50  == Number of False Negatives')

if st.checkbox('Conclusion'):
    st.text(f"Blood donation is a difficult task and sometimes hard to get people to donate.\nThis analysis helped to see the number of people who donate blood frequently in a certain month. \nA model was built with a Logistic Regression Classifier with Accuracy score of 0.7165. \nA confusion matrix was then used measure the performance of the test and train data.")
    



# Visualize Confusion Matrix

# viz = sns.heatmap(cm, annot=True)

# if st.checkbox('Visualize Confusion Matrix'):
    
#     st.pyplot(viz)
