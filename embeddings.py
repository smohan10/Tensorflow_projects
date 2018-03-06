# Import the libraries
import tensorflow as tf
import numpy as np 
import random
import argparse
import os
import sys
import shutil
import logging
import datetime as dt
import time
import emoji
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
import re

"""
The code is designed to train a classifier which detects emoji based on a sentence. 
The code iterates over the definition of embeddings explaining how glove vector
embeddings can represent words and relationship between words. 

The emoji classifier is split into 2 parts. 
Part I: Simple softmax classifier which takes in an input sentence, computes 
the average sentence embedding and outputs a class 0-4.

Part II: Demonstration of application of Sequence Modeling using LSTM and Keras
for building a Recurrent Model which elimates vanishing gradient problems in Part I.
"""


# Define the emoji dictionary
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}


def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def convert_to_one_hot(Y, dim=5):
    """
    Computes the one hot encoded vector of input matrix and dimension of the vectorization.
    """
    one_hot_vector = np.eye(dim)[Y.reshape(-1)]    
    return one_hot_vector
    

def read_csv(filename):
    """
    Given a CSV filename, reads the data into numpy array 
    """
    df = pd.read_csv(filename, header=None, names = ["phrase", "emoji"], usecols=[0,1])
    
    phrase_array = np.asarray(df["phrase"])
    emoji_array = np.asarray(df["emoji"], dtype=int)
    
    return phrase_array, emoji_array


def read_tsv(filename, file_type="train"):
    """
    Given a TSV file, reads the file and parses the inputs and outputs
    Used for sentiment analysis data, NOT coded completely in this code.
    """
    df = pd.read_csv(filename, sep='\t')
    
    
    init_id = df["SentenceId"][0]
    
    number_of_sentences_list = np.unique(df["SentenceId"])
    number_of_sentences = number_of_sentences_list[-1] - init_id + 1
    phrase_array =  []
    sentiment_array =  np.zeros(number_of_sentences,dtype=int)
    
    for i in range(number_of_sentences):
        idx = (df["SentenceId"]==i+1).argmax()        
        phrase_array.append(df.iloc[idx,:]["Phrase"].strip())
        
        if file_type=="train":
            sentiment_array[i] = df.iloc[idx,:]["Sentiment"]
        
    
    if file_type=="train":
        return np.asarray(phrase_array), sentiment_array
    else:
        return np.asarray(phrase_array)

    
def read_glove_vector_embedding(filename):
    """
    Given a filename, read the glove vector embedding into a dictionary
    Also prepare the mapping between word to index and vice-versa
    """
    
    with open(filename, "r", encoding="utf8") as gv:
        list_lines = gv.readlines()
        word_to_vector = {}
        words = set()
        for line in list_lines:
            split = line.split(" ")
            current_word = split[0].strip().lower()
            words.add(current_word)

            word_to_vector[current_word] = np.array(split[1:], dtype=np.float64)            
        
    i = 1
    word_to_idx = {}
    idx_to_word = {}
    
    for w in sorted(words):
        word_to_idx[w] = i
        idx_to_word[i] = w
        i += 1
        
    return words, word_to_vector, word_to_idx, idx_to_word


def cosine_similarity(embedding1, embedding2):
    """
    Compute the cosine similarity between 2 embedding vectors
    """
    numerator = np.dot(embedding1.T, embedding2)
    u_norm = np.sqrt(np.sum(np.square(embedding1)))
    v_norm = np.sqrt(np.sum(np.square(embedding2)))
    
    cosine_distance = numerator / (u_norm * v_norm)    
    return cosine_distance
    

def find_the_analogy(words_tuple, word_to_vector, words):
    """
    Given the relationship between 2 words in the tuple, find the analogy to the last
    word in the tuple. 
    Find the word that maximized the cosine similarity
    """
    max_similarity = -100    
    analogy_word = ""
    
    w1 = words_tuple[0].lower()
    w2 = words_tuple[1].lower()
    test_word = words_tuple[-1].lower()
    
    embed_1 = word_to_vector[w1]
    embed_2 = word_to_vector[w2]
    embed_test = word_to_vector[test_word]
        
    for w in words:
        if w in [w1, w2, test_word]:
            continue
            
        embed_w = word_to_vector[w]
        similarity = cosine_similarity(embed_test - embed_w, embed_1 - embed_2)
        if similarity > max_similarity:
            max_similarity = similarity
            analogy_word = w
    
    
    return analogy_word


def sentences_to_index(X, word_to_idx, max_len):
    """
    Convert the sentence to indexes. The sentence index vector is of len max_len
    """             
                
    m = len(X)
    sent_to_idx = np.zeros(shape=(m, max_len))

    for i in range(m):
        sentence = X[i]
        list_words = re.split(r'[\s-]+', sentence)
        
        for j in range(len(list_words)):
            current_word = list_words[j].lower()
            if current_word not in word_to_idx.keys():
                continue
            sent_to_idx[i, j] = word_to_idx[current_word]
               
    return sent_to_idx


def initialize_parameters(n_h, n_y):
    """
    Initialize the weights and biases
    """
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros(shape=(n_y,))
    
    return W, b


def compute_average_embedding(sentence, word_to_vector, embedding_dimension):
    """
    Given a sentence, compute the average embedding of size embedding_dimension
    """    
    average_embedding = np.zeros(embedding_dimension)
    
    sentence_list = sentence.strip().lower().split(" ")
    length = len(sentence_list)
    
    for word in sentence_list:
        if word == "": continue 
        average_embedding += word_to_vector[word]
        
    average_embedding = average_embedding/length
    
    return average_embedding
    

def compute_accuracy(actual_Y, predicted_Y):
    """
    Compute the accuracy of predicted Y values
    """
    total_samples = len(actual_Y)
    correct_pred = np.sum(np.equal(actual_Y, predicted_Y))
    
    return (correct_pred/total_samples)*100
    
    
def model(X, Y, word_to_vector,  embedding_dimension = 50, num_epochs = 1000, learning_rate=0.001):
    """
    Simple softmax model that takes in inputs X and Y, compute the average embedding per sample
    Predicts the y value and compute the loss. Finally gradients are calculated and weights 
    are updated. This is performed over all examples over all iterations.
    """
    np.random.seed(1)
    n_y = 5
    n_h = embedding_dimension
    m   = len(X)
    
    # Initialize the weights
    W, b = initialize_parameters(n_h, n_y)
    loss = 0 
    Y_one_hot = convert_to_one_hot (Y, n_y)
    
    
    for i in range(num_epochs):
        for j in range(m):
            
            # Compute the average embedding for the sentence
            current_embedding = compute_average_embedding(X[j], word_to_vector, embedding_dimension)
            
            # Compute the activation
            z = np.dot(W, current_embedding) + b
            
            # Compute the softmax of activation
            y_hat = softmax(z)
                        
            # Compute the loss
            loss = -1  * np.sum(Y_one_hot[j] * np.log(y_hat))
            
            # Compute the gradients
            dz = y_hat - Y_one_hot[j]            
            dW = np.dot(dz.reshape(n_y,1), current_embedding.reshape(1, embedding_dimension))
            db = dz
            
            # Update the parameters
            W = W - learning_rate * dW
            b = b - learning_rate * db
            
        if i % 100 == 0:
            print("Iteration: %d ----- Cost: %f" % (i, loss))
            
    params = (W, b)
    return params, loss



def predict_emoji(params, X, word_to_vector, embedding_dimension):
    """
    Given X and model parameters, predict the emoji
    """
    W, b = params[0], params[-1]
    
    m = len(X)
    Y_pred = np.zeros(m)
    
    for i in range(m):
        
        current_embedding = compute_average_embedding(X[i].strip(), word_to_vector, embedding_dimension)
            
        # Compute the activation
        z = np.dot(W, current_embedding) + b
            
        # Compute the softmax of activation
        y_hat = softmax(z)
        
        Y_pred[i] = np.argmax(y_hat)
    
    return Y_pred


def pre_trained_embedding_layer(word_to_vector, word_to_idx):
    """
    Compute an embedding layer for the given glove vector embedding matrix
    """
    vocab_len = len(word_to_idx) + 1
    emb_dim = word_to_vector["hi"].shape[0]
    
    emb_matrix = np.zeros(shape=(vocab_len, emb_dim))
    
    # Prepare the embedding matrix for each word index
    for word, index in word_to_idx.items():
        emb_matrix[index, :] = word_to_vector[word]
        
    # Create an embedding instance for input vocal dimension and output as embedding dimension    
    embedding = Embedding(input_dim=vocab_len, output_dim=emb_dim, embeddings_initializer='glorot_uniform', trainable=False)
    
    # Build the embedding layer before setting the weights
    embedding.build((None,))
    
    # Set the weights for the embedding layer
    embedding.set_weights([emb_matrix])
    
    return embedding
    
    
    
def keras_model(input_shape, word_to_vec_map, word_to_index):
    """
    Create a Keras model instance given the input shape. 
    The model consists of embedding layer, LSTM layer, Dropout, LSTM layer, Dropout, Dense layer and Softmax Activation.
    
    Architecture
    
    X => Embedding Layer => LSTM => Dropout => LSTM => Dropout => Dense Layer => Softmax => Y
    
    
    """
    
    # Create an input layer
    sentence_index = Input(shape=input_shape)
    
    # Create embedding layer
    embedding_layer = pre_trained_embedding_layer(word_to_vector, word_to_idx)
    
    # Pass the input to the embedding layer
    X = embedding_layer(sentence_index)
    
    # 1st LSTM layer
    X = LSTM(256, return_sequences=True)(X)
    
    # Dropout layer
    X = Dropout(rate=0.2)(X)
    
    # 2nd LSTM layer
    X = LSTM(256, return_sequences=False)(X)
    
    # Dropout
    X = Dropout(rate=0.2)(X)
    
    # Dense layer
    X = Dense(5, activation='softmax')(X)
    
    # Softmax Activation
    X = Activation('softmax')(X)
    
    # Create the Keras model 
    model = Model(input=[sentence_index], output=[X])
    
    return model
    
    
def run_training(X_train, Y_train_one_hot, X_test, Y_test_one_hot, word_to_vector, word_to_idx, num_epochs):
    
    """
    Run Sequence modeling for the inputs, train the model, compute the loss and accuracy and evaluate on the 
    test set.
    """
    
    max_len = np.max([len(sent.split(" ")) for sent in X_train])
    X_indexed_sentences = sentences_to_index(X_train, word_to_idx, max_len)
    X_test_indexed_sentences = sentences_to_index(X_test, word_to_idx, max_len)
    
    print("Max length of a sentence: %d" % max_len)
    
    
    
    lstm_model = keras_model((max_len,), word_to_vector, word_to_idx)
    
    lstm_model.summary()
        
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    lstm_model.fit(X_indexed_sentences,Y_train_one_hot,epochs=num_epochs)
    
    loss, acc = lstm_model.evaluate(X_indexed_sentences, Y_train_one_hot)    
    print("LSTM Model Training accuracy: %f" % acc)
    
    loss, acc = lstm_model.evaluate(X_test_indexed_sentences, Y_test_one_hot)    
    print("LSTM Model Test accuracy: %f" % acc)
    
    return lstm_model, max_len


# **************************************************************************************************************************

if __name__ == "__main__":

    
    print("*********************** PART I ***************************")
    
    
    # Read the glove vec
    filename = "data/glove.6B.50d.txt"
    print("Loading the embeddings file: %s" % filename)
    words, word_to_vector, word_to_idx, idx_to_word = read_glove_vector_embedding(filename)
    print("Extracted the word to vector")
    print("Number of words in the file: %d" % len(words))
    
    
    
    # Find the cosine similarity
    word1 = "he"
    word2 = "she"
    
    embedding1 = word_to_vector[word1.lower()]
    embedding2 = word_to_vector[word2.lower()]
    
    print("Similarity between %s and %s is %f" % (word1, word2, cosine_similarity(embedding1, embedding2)))
    
    # Find the anology
    print("Computing analogies...")
    
    words_tuple_list = [ ("italy", "italian" , "spain"), 
                         ("india", "delhi" , "japan"),
                         ("small", "smaller" , "large")
                       ]
    for word_tuple in words_tuple_list:        
        analogy_word = find_the_analogy(word_tuple, word_to_vector, words)
        print("Analogy:  %s --> %s :: %s --> %s" % (word_tuple[0], word_tuple[1], word_tuple[-1], analogy_word))
    
    
    
    """
    Part 2
    Emoji classifier
    
    Simple emoji classifier. Given a sentence and an appropriate label train a classifier, 
    predict emoji for the test sentence
    Compute the average embedding for a sentence, pass it through softmax classifier.    
    """
    print("\n*********************** PART II ***************************\n")
    
    X_train, Y_train = read_csv('data/train_emoji.csv')
    
    print("Size of training data: %d" % len(X_train))
    
    random_idx = np.random.randint(len(X_train))
    print("Example training phrase and emoji: %s ---> %d" % (X_train[random_idx], Y_train[random_idx]))
        
    X_test,  Y_test  = read_csv('data/tesss.csv')    
    print("Size of test data: %d" % len(X_test))
    
    random_idx = np.random.randint(len(X_test))
    print("Example test phrase and emoji: %s ---> %d" % (X_test[random_idx], Y_test[random_idx]))
   

    
    # Testing average embedding
    print("Sentence: %s" % X_test[random_idx])
    print("Average embedding: %r " % compute_average_embedding( X_test[random_idx], word_to_vector))
    
    # Testing one hot encoding
    test_vector = convert_to_one_hot(Y_test, 5)
    print("Index %d is converted to: %r" % (Y_test[random_idx],test_vector[random_idx] ))
     
    print("Convert labels into one hot encodings")
    Y_train_one_hot = convert_to_one_hot(Y_train, 5)
    Y_test_one_hot  = convert_to_one_hot(Y_test,  5)
    
    
    """
    Define the model and train
    
    """
    
    embedding_dimension = len(word_to_vector[list(word_to_vector.keys())[0]])
    
    
    print("BEGIN TRAINING....")
    params, loss = model( X_train, Y_train, word_to_vector, embedding_dimension, 1000, 0.001)
    
    
    print("PREDICT TRAINING ACCURACY...")
    predicted_Y = predict_emoji(params, X_train, word_to_vector, embedding_dimension)
    accuracy = compute_accuracy(Y_train, predicted_Y)
    
    print("Prediction accuracy: %f" % accuracy)
    
    print("PREDICT TEST ACCURACY...")
    predicted_Y = predict_emoji(params, X_test, word_to_vector, embedding_dimension)
    accuracy = compute_accuracy(Y_test, predicted_Y)
    
    print("Prediction accuracy: %f" % accuracy)
    
    
    print("\n*********************** PART III ***************************\n")
    
    """
    LSTM version for emoji classification   
    Define an embedding layer
    convert sentences into index vector m x maxLen
    """
    
    
    lstm_model, max_len = run_training(X_train, Y_train_one_hot, X_test, Y_test_one_hot, word_to_vector, word_to_idx, 250)
    
    
    # Test your own sentence: 
    test = np.asarray(["not feeling happy"])
    s_test = sentences_to_index(test, word_to_idx, max_len)
    print("LSTM MODEL Predicting for sentence: %s ----> %d" % (test[0], np.argmax(lstm_model.predict(s_test))))

    
    