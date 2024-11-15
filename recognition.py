# import tensorflow as tf 
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np 
# import pandas as pd 

# test_title = ["spark an inner revolution"]

# labels = ["Reliable", "Unreliable"]

# vocab_size = 5000
# paddingLen = 20
# oneHotRep = [one_hot(words, vocab_size) for words in test_title]
# padded = pad_sequences(oneHotRep, truncating="post", padding="post", maxlen=paddingLen)

# x = np.array(padded)

# model = load_model("fake_news.h5")

# pred = model.predict_classes(x)[0]
# print(labels[int(pred)])


# import tensorflow as tf 
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np 

# test_title = ["spark an inner revolution"]

# labels = ["Reliable", "Unreliable"]

# vocab_size = 5000
# paddingLen = 20
# oneHotRep = [one_hot(words, vocab_size) for words in test_title]
# padded = pad_sequences(oneHotRep, truncating="post", padding="post", maxlen=paddingLen)

# x = np.array(padded)

# model = load_model("fake_news.h5")

# # Manually set the input_length parameter for the Embedding layer
# model.layers[0].input_length = paddingLen

# pred = model.predict_classes(x)[0]
# print(labels[int(pred)])


# import tensorflow as tf 
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np 

# test_title = ["spark an inner revolution"]

# labels = ["Reliable", "Unreliable"]

# vocab_size = 5000
# paddingLen = 20
# oneHotRep = [one_hot(words, vocab_size) for words in test_title]
# padded = pad_sequences(oneHotRep, truncating="post", padding="post", maxlen=paddingLen)

# x = np.array(padded)

# # Load the model without compiling it
# model = load_model("new_model.h5", compile=False)

# # Manually set the input_length parameter for the Embedding layer
# model.layers[0].input_length = paddingLen

# # Compile the model after setting the input_length
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# # Predict using the loaded and compiled model
# pred = model.predict_classes(x)[0]
# print(labels[int(pred)])



## -----------------------------------------------Working Code ##
# import tensorflow as tf 
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np 

# test_title = ["Breaking: Alien invasion imminent"]

# labels = ["Reliable", "Unreliable"]

# vocab_size = 5000
# paddingLen = 20
# oneHotRep = [one_hot(words, vocab_size) for words in test_title]
# padded = pad_sequences(oneHotRep, truncating="post", padding="post", maxlen=paddingLen)

# x = np.array(padded)

# # Load the model without compiling it
# model = load_model("new_model.h5", compile=False)

# # Manually set the input_length parameter for the Embedding layer
# model.layers[0].input_length = paddingLen

# # Compile the model after setting the input_length
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# # Predict using the loaded and compiled model
# y_pred_prob = model.predict(x)
# y_pred = (y_pred_prob > 0.5).astype("int32")

# print(labels[y_pred[0][0]])


import pickle
import streamlit as st
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 

# Load evaluation results
with open('evaluation_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Function for preprocessing text
def preprocess_text(text):
    # Your preprocessing steps here
    return text

# Load the model
model = load_model("new_model.h5", compile=False)

# Define labels
labels = ["Reliable", "Unreliable"]

# Define vocabulary size and padding length
vocab_size = 5000
padding_len = 20

# Streamlit app
def main():
    st.title("Fake News Detection")
    st.write("Enter a title to detect whether it's reliable or unreliable.")
    print("Hello")
    # Input field for the title
    title = st.text_input("Title", "")

    # Predict button
    if st.button("Predict"):
        if title:
            # Preprocess the title
            processed_title = preprocess_text(title)

            # One-hot encode and pad the title
            one_hot_rep = [one_hot(processed_title, vocab_size)]
            padded = pad_sequences(one_hot_rep, truncating="post", padding="post", maxlen=padding_len)

            # Convert to numpy array
            x = np.array(padded)

            # Predict using the loaded and compiled model
            y_pred_prob = model.predict(x)
            y_pred = (y_pred_prob > 0.5).astype("int32")

            # Display prediction
            prediction_label = labels[y_pred[0][0]]
            st.write("Prediction:", prediction_label)
        else:
            st.write("Please enter a title.")

    st.write("Accuracy:", results['accuracy'])
    st.write("Confusion Matrix:")
    st.write(results['confusion_matrix'])
    st.write("Classification Report:")
    st.write(results['classification_report'])


main()