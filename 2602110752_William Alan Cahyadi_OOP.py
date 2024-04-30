#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

class DataImporter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_frame = None
        self.features = None
        self.target = None

    def load_data(self, delimiter=','):
        self.data_frame = pd.read_csv(self.file_path, delimiter=delimiter)

    def separate_features_target(self, target_column):
        self.target = self.data_frame[target_column]
        self.features = self.data_frame.drop(target_column, axis=1)

class DataProcessor:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.features_train = None
        self.features_test = None
        self.target_train = None
        self.target_test = None

    def remove_columns(self, columns_to_drop):
        self.features.drop(columns=columns_to_drop, inplace=True)

    def split_data(self, test_ratio=0.2, seed=42):
        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(self.features, self.target, test_size=test_ratio, random_state=seed)

    def fill_na(self, strategy='mean', column=None):
        if strategy == 'mean':
            replacement_value = self.features_train[column].mean()
        elif strategy == 'median':
            replacement_value = self.features_train[column].median()
        elif strategy == 'mode':
            replacement_value = self.features_train[column].mode()[0]
        self.features_train[column].fillna(replacement_value, inplace=True)
        self.features_test[column].fillna(replacement_value, inplace=True)

    def encode_categorical(self, column):
        self.features_train = pd.get_dummies(self.features_train, columns=[column])
        self.features_test = pd.get_dummies(self.features_test, columns=[column])

    def scale_features(self, method='robust', columns=None):
        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        self.features_train[columns] = scaler.fit_transform(self.features_train[columns])
        self.features_test[columns] = scaler.transform(self.features_test[columns])

class ModelTrainer:
    def __init__(self, features_train, target_train, features_test, target_test):
        self.features_train = features_train
        self.target_train = target_train
        self.features_test = features_test
        self.target_test = target_test
        self.model = None

    def train_model(self):
        self.model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth=3, n_estimators=200, random_state=42)
        self.model.fit(self.features_train, self.target_train)

    def evaluate(self):
        predictions = self.model.predict(self.features_test)
        print('\nClassification Report\n')
        print(classification_report(self.target_test, predictions))

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    # load data
    importer = DataImporter('data_C.csv')
    importer.load_data()
    importer.separate_features_target('churn')

    # data processor and preprocess
    processor = DataProcessor(importer.features, importer.target)
    processor.remove_columns(['Unnamed: 0', 'id', 'CustomerId', 'Surname'])
    processor.split_data()
    processor.fill_na(column='CreditScore')
    processor.encode_categorical('Geography')
    processor.scale_features(columns=['Age', 'CreditScore', 'Balance', 'EstimatedSalary'])

    # train and evaluate
    trainer = ModelTrainer(processor.features_train, processor.target_train, processor.features_test, processor.target_test)
    trainer.train_model()
    trainer.evaluate()

    # trainer.save_model('best_model_xgb.pkl')