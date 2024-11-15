# import nltk
# import tensorflow as tf 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Dense
# from tensorflow.keras.layers import LSTM, Activation, SpatialDropout1D
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import matplotlib.pyplot as plt 
# import pandas as pd 
# import numpy as np 
# import re

# train_dir = "data/train.csv"
# df = pd.read_csv(train_dir)

# df = df.dropna()
# df = df.reset_index()
# X = df.drop(labels=['label', 'id'], axis=1)
# y = df['label']

# xdata = X.copy()
# xdata.reset_index(inplace=True)

# lemmatizer = WordNetLemmatizer()
# stop_words = stopwords.words('english')

# xtitle = []
# for i in range(len(xdata)):
# 	sent = re.sub('[^a-zA-Z]', ' ', xdata['title'][i])
# 	sent = sent.lower().split()
# 	sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stop_words)]
# 	sent = " ".join(sent)
# 	xtitle.append(sent)

# vocab_size = 5000
# embedding_feature_len = 30
# max_sent_len = 20
# batch_size = 32
# epochs = 10

# one_hot_representation = [one_hot(words, vocab_size) for words in xtitle]
# padded_sequences = pad_sequences(one_hot_representation, truncating="post",
# 				padding="post", maxlen=max_sent_len)

# X = np.array(padded_sequences)
# y = np.array(y)

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = Sequential()
# model.add(Embedding(vocab_size, embedding_feature_len, input_length=max_sent_len))
# model.add(SpatialDropout1D(rate=0.2))
# model.add(LSTM(units=128))
# model.add(Dense(units=1))
# model.add(Activation("sigmoid"))
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# print(model.summary())

# hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# y_pred = model.predict_classes(x_test)





# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# import tensorflow as tf 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Dense
# from tensorflow.keras.layers import LSTM, Activation, SpatialDropout1D
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import matplotlib.pyplot as plt 
# import pandas as pd 
# import numpy as np 
# import re

# # Lemmatizer and stopwords initialization
# lemmatizer = WordNetLemmatizer()
# stop_words = stopwords.words('english')

# # Some constants
# vocab_size = 5000
# embedding_feature_len = 30
# max_sent_len = 20
# batch_size = 32
# epochs = 10

# # Example titles data
# xdata = pd.DataFrame({"title": ["This is an example sentence", "Another example sentence"]})

# # Preprocessing the text
# xtitle = []
# for i in range(len(xdata)):
#     sent = re.sub('[^a-zA-Z]', ' ', xdata['title'][i])
#     sent = sent.lower().split()
#     sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stop_words)]
#     sent = " ".join(sent)
#     xtitle.append(sent)

# # One-hot encoding and padding
# one_hot_representation = [one_hot(words, vocab_size) for words in xtitle]
# padded_sequences = pad_sequences(one_hot_representation, truncating="post",
#                 padding="post", maxlen=max_sent_len)

# # Converting data to numpy arrays
# X = np.array(padded_sequences)

# # Defining a sample label array
# y = np.array([0, 1])

# # Splitting data into train and test sets
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Defining the model architecture
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_feature_len, input_length=max_sent_len))
# model.add(SpatialDropout1D(rate=0.2))
# model.add(LSTM(units=128))
# model.add(Dense(units=1))
# model.add(Activation("sigmoid"))
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# print(model.summary())

# # Training the model
# hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# # Making predictions
# y_pred = model.predict_classes(x_test)


## -----------------------------------------------Working Code ##
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# import tensorflow as tf 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Dense
# from tensorflow.keras.layers import LSTM, Activation, SpatialDropout1D
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import matplotlib.pyplot as plt 
# import pandas as pd 
# import numpy as np 
# import re

# # Lemmatizer and stopwords initialization
# lemmatizer = WordNetLemmatizer()
# stop_words = stopwords.words('english')

# # Some constants
# vocab_size = 5000
# embedding_feature_len = 30
# max_sent_len = 20
# batch_size = 32
# epochs = 10

# # Example titles data
# xdata = pd.DataFrame({"title": ["This is an example sentence", "Another example sentence"]})

# # Preprocessing the text
# xtitle = []
# for i in range(len(xdata)):
#     sent = re.sub('[^a-zA-Z]', ' ', xdata['title'][i])
#     sent = sent.lower().split()
#     sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stop_words)]
#     sent = " ".join(sent)
#     xtitle.append(sent)

# # One-hot encoding and padding
# one_hot_representation = [one_hot(words, vocab_size) for words in xtitle]
# padded_sequences = pad_sequences(one_hot_representation, truncating="post",
#                 padding="post", maxlen=max_sent_len)

# # Converting data to numpy arrays
# X = np.array(padded_sequences)

# # Defining a sample label array
# y = np.array([0, 1])

# # Splitting data into train and test sets
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Defining the model architecture
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_feature_len))
# model.add(SpatialDropout1D(rate=0.2))
# model.add(LSTM(units=128))
# model.add(Dense(units=1))
# model.add(Activation("sigmoid"))
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# print(model.summary())

# # Training the model
# hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# # Making predictions
# y_pred_prob = model.predict(x_test)
# y_pred = (y_pred_prob > 0.5).astype("int32")

# # Save the model
# model.save("new_model.h5")

import pickle
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import LSTM, Activation, SpatialDropout1D
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np

# Lemmatizer and stopwords initialization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Some constants
vocab_size = 5000
embedding_feature_len = 30
max_sent_len = 20
batch_size = 32
epochs = 10

# Read CSV file
data = pd.read_csv('train.csv')

data.dropna(subset=['title'], inplace=True)

# Preprocessing the text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    text = " ".join(text)
    return text

data['processed_text'] = data['title'].apply(preprocess_text)

# One-hot encoding and padding
one_hot_representation = [one_hot(words, vocab_size) for words in data['processed_text']]
padded_sequences = pad_sequences(one_hot_representation, truncating="post",
                padding="post", maxlen=max_sent_len)

# Converting data to numpy arrays
X = np.array(padded_sequences)
y = np.array(data['label'])

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the model architecture
model = Sequential()
model.add(Embedding(vocab_size, embedding_feature_len))
model.add(SpatialDropout1D(rate=0.2))
model.add(LSTM(units=128))
model.add(Dense(units=1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

# Training the model
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Making predictions
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Save the model
model.save("new_model.h5")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

results = {}

# Evaluate and store metrics
results['accuracy'] = accuracy_score(y_test, y_pred)
results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)

# Print the results
for metric, value in results.items():
    print(metric + ':', value)
    
# Save results as a Python object
with open('evaluation_results.pkl', 'wb') as f:
    pickle.dump(results, f)