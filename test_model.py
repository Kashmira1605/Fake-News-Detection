# import pandas as pd
# import numpy as np
# import tensorflow as tf 
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# from fake_news_model import preprocess_text, vocab_size, max_sent_len


# # Assuming you have already trained and saved the model as 'new_model.h5'


# # Load the test dataset
# test_data = pd.read_csv('test.csv')  # Replace 'your_test_csv_file.csv' with the path to your test dataset

# # Preprocess the text data in the test dataset
# test_data['processed_text'] = test_data['title'].apply(preprocess_text)

# # Encode the preprocessed text data
# test_one_hot_representation = [one_hot(words, vocab_size) for words in test_data['processed_text']]
# test_padded_sequences = pad_sequences(test_one_hot_representation, truncating="post",
#                 padding="post", maxlen=max_sent_len)

# # Converting data to numpy arrays
# X_test = np.array(test_padded_sequences)
# y_test = np.array(test_data['label'])

# # Load the trained model
# loaded_model = tf.keras.models.load_model("new_model.h5")

# # Use the loaded model to make predictions on the test data
# y_pred_prob_test = loaded_model.predict(X_test)
# y_pred_test = (y_pred_prob_test > 0.5).astype("int32")

# # Evaluate the performance of the model
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# accuracy = accuracy_score(y_test, y_pred_test)
# conf_matrix = confusion_matrix(y_test, y_pred_test)
# class_report = classification_report(y_test, y_pred_test)

# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", class_report)


import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from fake_news_model import preprocess_text, vocab_size, max_sent_len

# Load the test dataset
test_data = pd.read_csv('test.csv')  # Replace 'your_test_csv_file.csv' with the path to your test dataset
test_data.dropna(subset=['title'], inplace=True)

# Preprocess the text data in the test dataset
test_data['processed_text'] = test_data['title'].apply(preprocess_text)

# Encode the preprocessed text data
test_one_hot_representation = [one_hot(words, vocab_size) for words in test_data['processed_text']]
test_padded_sequences = pad_sequences(test_one_hot_representation, truncating="post",
                padding="post", maxlen=max_sent_len)

# Converting data to numpy arrays
X_test = np.array(test_padded_sequences)
y_test = np.array(test_data['label'])

# Load the trained model
loaded_model = tf.keras.models.load_model("new_model.h5")

# Use the loaded model to make predictions on the test data
y_pred_prob_test = loaded_model.predict(X_test)
y_pred_test = (y_pred_prob_test > 0.5).astype("int32")

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)