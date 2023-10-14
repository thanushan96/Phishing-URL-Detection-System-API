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


# df = pd.read_csv('NLP_Dataset.csv')
# x= df['url']
# y = df['label']

# voc_size = 10000
# messages = x.copy()

# corpus = []
# for i in range(0, len(messages)):
#     review = re.sub('[^a-zA-Z]',' ',urlparse(messages[i]).netloc)
#     review = review.lower()
#     review = review.split()
#     review=' '.join(review)
#     corpus.append(review)

# onehot_repr=[one_hot(words,voc_size)for words in corpus]
# sent_length = 50
# embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

# embedded_docs = np.array(embedded_docs)

# #x_final = np.array(embedded_docs)
# x_final = embedded_docs
# y_final  = np.array(y)
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.20,random_state=15)


# #make the model and train it
# embedding_vector_features=100
# model = Sequential()
# model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
# model.add(LSTM(100))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=2,batch_size=64)

# y_pred1=model.predict(x_test) 
# classes_y1=np.round(y_pred1).astype(int)
# from sklearn.metrics import confusion_matrix
# confusion_n = confusion_matrix(y_test,classes_y1)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, classes_y1))
# model.save("model_NLP.h5")
#%%
from flask import Flask
from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS  
import database
from routes.auth import auth_bp
from routes.predict import predict_bp

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": "http://localhost:3000"},
    r"/login": {"origins": "http://localhost:3000"},
    r"/register": {"origins": "http://localhost:3000"}
})
# # Load NLP and feature-based models
# model = tf.keras.models.load_model('model_NLP.h5')
# modelfeature = tf.keras.models.load_model('modelGRU.h5')
#%%



app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/DB'
app.secret_key = 'c3c276452481cef3fae39c2de9475431f1c5ea781dad07be097be89fe0666e09'
# Initialize the database  
database.initialize_database(app)


#%%
# Register the blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)

#%%


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
