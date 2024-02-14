
import streamlit as st
import pickle
import pandas as pd

st.title('Titanic Survival Prediction')

# Read Data
df = pd.read_csv('titanic_train.csv')
df = df.dropna()

# Take inputs
pclass = st.selectbox('Pclass', df.Pclass.unique())
sex = st.selectbox('Sex', df.Sex.unique())
age = st.number_input('Age')
sibsp = st.number_input('SibSp', df.SibSp.min(), df.SibSp.max())
parch = st.number_input('Parch', df.Parch.min(), df.Parch.max())
fare = st.number_input('Fare', df.Fare.min(), df.Fare.max())
embarked = st.selectbox('Embarked', df.Embarked.unique())

# Data Format
data = {
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}

model = pickle.load(open('pipeline.pkl', 'rb'))
pred = model.predict(pd.DataFrame(data, index=[0]))

if st.button('Predict'):
    if pred[0] == 0:
        st.error('Not survived')
    else:
        st.success('Survived')
