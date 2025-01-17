import streamlit as st
import pandas as pd
import pickle

st.title('Predict Car Price ')

df=pd.read_csv('dataset/updated_quikr_car.csv')

name_option=list(df['name'].unique())
company_option=list(df['company'].unique())
fuel_type_option=list(df['fuel_type'].unique( ))


name=st.selectbox("Name",name_option)
company=st.selectbox("Company",company_option)
year=st.slider("Year",min_value=2000,max_value=2020,step=1)

kms_driven=st.slider("Kms Driven",min_value=0,max_value=200000,step=1)

fuel_type=st.selectbox("Fuel type",fuel_type_option)

pred=pd.DataFrame({'name':[name],'company':[company],'year':[year],'kms_driven':[kms_driven],'fuel_type':[fuel_type]})

with open('model.pkl','rb') as file:
    model=pickle.load(file)


if st.button("Predict "):
    price=model.predict(pred)
    st.write(f'Predicted Price of the car {price}')
