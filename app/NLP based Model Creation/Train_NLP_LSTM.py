import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from urllib.parse import urlparse, urlencode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

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


x_final = embedded_docs
y_final  = np.array(y)
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
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("model_NLP.h5")

import re
from urllib.parse import urlparse 
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Replace 'unseen_url' with the URL you want to test
unseen_url = 'https://krasnoarmeyskaya.dipmarket.by/apj/one%20drive%20need%20fix(1)'

# Preprocess the unseen URL
review = re.sub('[^a-zA-Z]', ' ', urlparse(unseen_url).netloc)
review = review.lower()
review = review.split()
review = ' '.join(review)

# Convert the preprocessed URL to numeric representation
onehot_repr = [one_hot(review, voc_size)]  # Use the same voc_size as in your training

# Pad the sequence to match the input length used during training
padded_sequence = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

# Make predictions
raw_predictions = model.predict(padded_sequence)

# Convert raw predictions to binary labels
binary_labels = np.round(raw_predictions).astype(int)

# Print the result
if binary_labels[0] == 1:
    print(f"The URL '{unseen_url}' is predicted to be a phishing class.")
else:
    print(f"The URL '{unseen_url}' is predicted to be a legtimate class.")
