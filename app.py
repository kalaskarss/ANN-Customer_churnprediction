import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Loading the trained model
model = tf.keras.models.load_model(r'C:\\Udemy_DL\\model.h5')


#Load the trained model,scaler pickle,onhot

model= load_model(r'C:\\Udemy_DL\\model.h5')

#load the encoder and scaler

with open('Onehot_encode_geo.pkl','rb') as file:
    Onehot_encode_geo =pickle.load(file)

with open('label_encoder_gen.pkl','rb') as file:
    label_encoder_gen = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler= pickle.load(file)


# Streamlit app
st.title('Customer Churn Prediction')
st.image(r"C:\\Users\\LENOVO\\OneDrive\\Pictures\\chust.im.png")

st.markdown("""
<style>
input[type="text"] {
    width: 100%;
    height: 50px;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)




geography = st.selectbox('Geography', Onehot_encode_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gen.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# one hot encode 'Geography'ArithmeticError
geo_encoded =  Onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns= Onehot_encode_geo.get_feature_names_out(['Geography']))


# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
