## Group_11  - Question 3 (Sentiment Analysis)

# import required packages
import pickle
from glob import glob
import os,re,string
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding,Conv1D,MaxPooling1D,Flatten
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Extracting data from the imdb files
print("Loading the training dataset:")
def extract_text_from_files(path,folder_name):
  text_data = []
  labels = []
  for label_attached,label_given in enumerate(folder_name):
    for file_name in glob(os.path.join(path, label_given, '*.*')):
        text_data.append(open(file_name, 'r').read())
        labels.append(label_attached)

  return text_data, np.array(labels)

# Preprocessing and Removing special characters from the reviews using regex
print("Preprocessing and Removing special characters from the reviews using regex:")
def data_preprocessing(input_data):
    regex_char = re.compile("[.;:!#\'?,\"()\[\]]|(<br\s*/><br\s*/>)|(\-)|(\/)")
    
    return [regex_char.sub("", text.lower()) for text in input_data]

# List of stopwords
list_stopwords=set(stopwords.words('english'))
print("Stopwords:",list_stopwords)

# Removal of unwanted words as they do not provide valuable information for downstream analysis.
def remove_stopwords(data):
    final_list=[]
    for lines in data:
        temp_list=[]
        for word in lines.split():
            if word.lower() not in list_stopwords:
                temp_list.append(word)
        final_list.append(' '.join(temp_list))
    return final_list    

# Main method to perform the evaluation on the Imdb dataset provided
if __name__ == "__main__":

     # 1. Loading the training data
     print("load the training data :")
     X_train, y_train = extract_text_from_files(f'./data/aclImdb/train',['neg','pos'])
     
     # Data preprocessing
     X_train = data_preprocessing(X_train) 

     #Removing stopwords
     print("Stopwords Removal :")
     X_train = remove_stopwords(X_train)
    
     # Initializing the tokenizer
     print("Initializing the tokenizer:")
     tokenizer = Tokenizer()

     # Generating word indexes
     print("Generating word indexes:")
     tokenizer.fit_on_texts(X_train)

     #Storing the tokenizer fitted on the training data in a pickle file to be reused to perform evaluation on the test set
     file=pickle.dump(tokenizer,open(r"./models/Group_11_token.pkl","wb"))

     # Generate sequences for the training data
     print("Generate sequences:")
     X_train_N = tokenizer.texts_to_sequences(X_train)

     #Apply Padding to make the length of all sentences similar for better performance of the model
     print("Padding Sequence:")
     X_train_F = pad_sequences(X_train_N, padding='post',maxlen=1000)

     #Splitting the train dataset into train and validation sets for training
     X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_train_F, y_train, test_size=0.3 ,random_state=42)


# 2. Train your network
# Make sure to print your training loss and accuracy within training to show progress
# Make sure you print the final training accuracy

#Size of the words generated from tokenizer
size_word = len(tokenizer.word_index)+1

model_NN = keras.Sequential()
model_NN.add(keras.layers.Embedding(size_word, 16))
model_NN.add(keras.layers.Dropout(0.25))
model_NN.add(keras.layers.Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))
model_NN.add(keras.layers.GlobalAveragePooling1D())
model_NN.add(keras.layers.Dropout(0.25))
model_NN.add(keras.layers.Dense(256, activation='relu'))
model_NN.add(keras.layers.Dropout(0.25))
model_NN.add(keras.layers.Dense(1, activation='sigmoid'))
model_NN.summary()

#Compiling the model created
model_NN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the model on the training data and evaluating the loss for the model based on the validation data
history = model_NN.fit(tf.convert_to_tensor(X_train_data),tf.convert_to_tensor(y_train_data),epochs=10,validation_data=(tf.convert_to_tensor(X_val_data), tf.convert_to_tensor(y_val_data)),verbose=1,batch_size=512)
print('\nThe final training accuracy is ',history.history['accuracy'][-1]*100)

print('\nThe final validation accuracy is ',history.history['val_accuracy'][-1]*100)

# 3. Saving the model for evaluating the test set
model_NN.save("./models/Group11_NLP_model.h5")

#Plotting the loss and accuracy obtained for the model
#fig = plt.figure(figsize=(18, 8))

#plt.subplot(1,2, 1)
#plt.plot(history.history['accuracy'],marker = '*',color='purple')
#plt.plot(history.history['val_accuracy'],marker = '*',color='red')
#plt.title("Accuracy v/s Epochs")
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
#plt.legend(['train', 'val'], loc='upper left')

#plt.subplot(1,2, 2)
#plt.plot(history.history['loss'],marker = '*',color='purple')
#plt.plot(history.history['val_loss'],marker = '*',color='red')
#plt.title("Loss v/s Epochs")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.legend(['train', 'val'], loc='upper left')

#plt.show()