import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, concatenate
from sklearn.model_selection import train_test_split


df1 = pd.read_csv('FeaturedDataset.csv')
#print(df.head())
df = df1.drop(['url'],axis=1).copy()

x = df.drop('label',axis=1)
y = df['label']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train=x_train.values.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.values.reshape(x_test.shape[0],x_test.shape[1],1)

# Create LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(14, 1)))
lstm_model.add(LSTM(50, return_sequences=True))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Create GRU model
gru_model = Sequential()
gru_model.add(GRU(50, return_sequences=True, input_shape=(14, 1)))
gru_model.add(GRU(50, return_sequences=True))
gru_model.add(GRU(50))
gru_model.add(Dense(1))
gru_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Define an input layer for combining models
input_layer = Input(shape=(14, 1))

# Get the outputs from LSTM and GRU models
lstm_output = lstm_model(input_layer)
gru_output = gru_model(input_layer)

# Concatenate the outputs from both models
concatenated_output = concatenate([lstm_output, gru_output], axis=-1)

# Add additional layers for predictions
ensemble_model = Dense(32, activation='relu')(concatenated_output)
ensemble_model = Dense(1)(ensemble_model)  # Output layer

# Create the final stacked model
stacked_model = Model(inputs=input_layer, outputs=ensemble_model)
stacked_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
stacked_model.summary()

# Train the stacked model

stacked_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=16, verbose=1)

# Evaluate the stacked model
y_pred_stacked = stacked_model.predict(x_test)
classes_y_stacked = np.round(y_pred_stacked).astype(int)

# Evaluate the stacked model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred_stacked = stacked_model.predict(x_test)
classes_y_stacked = np.round(y_pred_stacked).astype(int)

accuracy = accuracy_score(y_test, classes_y_stacked)
precision = precision_score(y_test, classes_y_stacked)
recall = recall_score(y_test, classes_y_stacked)
f1 = f1_score(y_test, classes_y_stacked)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, classes_y_stacked)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


# Save the stacked model
stacked_model.save('stacked_model.h5')

