from operator import index
from os import write
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write(''' # Iris Flower Prediction App
   This app predicts the ***Iris Flower Type!*** ''')

st.sidebar.header('User ***Input Parameters***')

def user_input_param():
    sepal_length = st.sidebar.slider(label='Sepal Length', min_value=4.3, max_value=7.9, value=5.4)
    sepal_width = st.sidebar.slider(label='Sepal Width', min_value=2.0, max_value=4.4, value=3.4)
    petal_length = st.sidebar.slider(label='Petal Length', min_value=1.0, max_value=6.9, value=1.3)
    petal_width = st.sidebar.slider(label='Petal Width', min_value=0.1, max_value=2.5, value=0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data=data, index=[0])
    return features

df = user_input_param()

st.subheader(body='User Input Parameters')
st.write(df)

iris_dataset = datasets.load_iris()
X = iris_dataset.data
Y = iris_dataset.target

model = RandomForestClassifier()
model.fit(X=X, y=Y)

prediction = model.predict(X=df)
prediction_prob = model.predict_proba(X=df)

st.subheader('***Class Labels*** name according to their index number')
for ind, target_name in enumerate(iterable=iris_dataset.target_names):
    st.write(f"Index {ind}: {target_name}")

st.subheader(body='Prediction')
st.write(f'Prediction: {prediction[0]}, Type: ***{iris_dataset.target_names[prediction][0]}***')


st.subheader('Prediction Probability')
st.write(prediction_prob)