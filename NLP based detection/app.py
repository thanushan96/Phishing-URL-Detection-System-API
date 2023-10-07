import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from urllib.parse import urlparse, urlencode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import whois
import ipaddress
import csv
from datetime import datetime
import requests
import ssl, socket
from flask import jsonify
import json
from domain_info import domaincreatedate, domainexpiredate, ageofdomain1
from feature_extraction import extract_features

import certificate_info
#%%


df = pd.read_csv('NLP_Dataset.csv')
x= df['url']
y = df['label']

voc_size = 10000
messages = x.copy()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',urlparse(messages[i]).netloc)
    review = review.lower()
    review = review.split()
    review=' '.join(review)
    corpus.append(review)

onehot_repr=[one_hot(words,voc_size)for words in corpus]
sent_length = 50
embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

embedded_docs = np.array(embedded_docs)

#x_final = np.array(embedded_docs)
x_final = embedded_docs
y_final  = np.array(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.20,random_state=15)


#make the model and train it
embedding_vector_features=100
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=2,batch_size=64)

y_pred1=model.predict(x_test) 
classes_y1=np.round(y_pred1).astype(int)
from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y_test,classes_y1)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, classes_y1))
model.save("model_NLP.h5")
#%%
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
# Load NLP and feature-based models
model = tf.keras.models.load_model('model_NLP.h5')
modelfeature = tf.keras.models.load_model('stacked_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    voc_size = 10000
    corpus = []
    classes_y = ''
    status = ''
    showmsg1 = ''
    a = ''
    b = ''
    urlstatus = ''
    for i in request.form.values():
        messages = i
        
        # Check if the URL is in valid format
        messages = urlparse(messages).netloc
        if messages == '':
            status = "Enter URL in valid format (e.g., 'https://www.example.com')"
            a = 1
        else:
            status = "URL is in valid format:"
            a = 2
        
        # Reading the whitelist
        f = open("whitelist.txt", "r")
        if messages in f.read():
            showmsg1 = "This is legitimate"
            b = 1
        else:
            showmsg1 = "Whitelist does not have any record!"
            b = 2
            
        review = re.sub('[^a-zA-Z]', ' ', messages)
        review = review.lower()
        review = review.split()
        review = ' '.join(review)
        corpus.append(review)
        onehot_repr = [one_hot(words, voc_size) for words in corpus]
        sent_length = 50
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
        x_test = embedded_docs
        y_pred = model.predict(x_test)
        classes_y = int(np.round(y_pred))
        
        # Predict using the feature-based trained model
        pf1 = pd.read_csv('test1.csv')
        pf = pf1.drop(['url'], axis=1).copy()
        x = pf.values.reshape(1, 14, 1)
        y = modelfeature.predict(x)
        
        # Define a threshold for NLP model prediction
        nlp_threshold = 0.6
        
        # Check if either of the models classify the URL as phishing
        if y_pred > nlp_threshold or y == 1:
            urlstatus = "Phishing URL"
        else:
            urlstatus = "Legitimate URL"
        
        # Retrieve certificate information using the certificate_info module
        issued_to = certificate_info.get_certificate_issued_to(messages)
        issued_by = certificate_info.get_certificate_issued_by(messages)
        
        # Create a dictionary to store response data
        response_data = {
            "status_value": status,
            "hidden_msg": showmsg1,
            "prediction_text": float(y_pred[0][0]),
            "prediction_text1": classes_y,
            "urlstatus": urlstatus,
            "URL_issued_by": issued_by,
            "URL_issued_to": issued_to,
            "created_date": domaincreatedate(messages),
            "expired_date": domainexpiredate(messages),
            "domain_age": ageofdomain1(messages),
            "featurebase_predict": float(y[0][0])
        }
        
        # If the input was invalid, reset values
        if a == 1:
            showmsg1 = ''
            y_pred = ''
            classes_y = ''
            y = ''
            urlstatus = ''
        
        return jsonify(response_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
