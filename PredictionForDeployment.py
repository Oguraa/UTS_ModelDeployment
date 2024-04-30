#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle

class ModelLoader:
    def __init__(self):
        self.models = {}

    def load_model(self, filepath):
        with open(filepath, 'rb') as file:
            self.models[filepath] = pickle.load(file)
        return self.models[filepath]

class DataScaler:
    def __init__(self, filepath):
        self.scaler = self.load_scaler(filepath)

    def load_scaler(self, filepath):
        with open(filepath, 'rb') as file:
            scaler = pickle.load(file)
        return scaler

    def scale_data(self, data, columns):
        return self.scaler.transform(data)[0]

class Predictor:
    def __init__(self, model, gender_encoder, geo_encoder, robust_scaler, minmax_scaler):
        self.model = model
        self.gender_encoder = gender_encoder
        self.geo_encoder = geo_encoder
        self.robust_scaler = robust_scaler
        self.minmax_scaler = minmax_scaler

    def preprocess(self, credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
        gender_encoded = self.gender_encoder["Gender"][gender]
        has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
        is_active_member_encoded = 1 if is_active_member == "Yes" else 0
        geo_encoded = self.geo_encoder.transform([[geography]]).toarray()[0]

        age_scaled, credit_score_scaled = self.robust_scaler.scale_data([[age, credit_score]], ['Age', 'CreditScore'])
        balance_scaled, estimated_salary_scaled = self.minmax_scaler.scale_data([[balance, estimated_salary]], ['Balance', 'EstimatedSalary'])

        features = pd.DataFrame({
            'CreditScore': [credit_score_scaled],
            'Gender': [gender_encoded],
            'Age': [age_scaled],
            'Tenure': [tenure],
            'Balance': [balance_scaled],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card_encoded],
            'IsActiveMember': [is_active_member_encoded],
            'EstimatedSalary': [estimated_salary_scaled]
        })

        geo_df = pd.DataFrame({f'Geography_{col}': [geo] for col, geo in zip(['France', 'Germany', 'Spain'], geo_encoded)})
        features = pd.concat([features, geo_df], axis=1)

        return features

    def predict(self, features):
        prediction = self.model.predict(features.values.reshape(1, -1))
        return 'Not Churn' if prediction[0] == 0 else 'Churn'

def main():
    st.title('Churn Prediction App')

    loader = ModelLoader()
    model = loader.load_model('BestModel_XGB_rev.pkl')
    gender_encoder = loader.load_model('gender_encode.pkl')
    geo_encoder = loader.load_model('oneHot_encode_geo.pkl')
    robust_scaler = DataScaler('robust_scaler.pkl')
    minmax_scaler = DataScaler('minmax_scaler.pkl')

    predictor = Predictor(model, gender_encoder, geo_encoder, robust_scaler, minmax_scaler)

    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100)
    tenure = st.number_input("Tenure", min_value=0, max_value=100)
    balance = st.number_input("Balance", min_value=0, max_value=100000)
    num_of_products = st.number_input("Number of Products", min_value=0, max_value=10)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0,
