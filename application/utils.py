#%%
import tensorflow as tf
from tensorflow.keras import layers  
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import LabelEncoder
from typing import List
import pickle
import os
import pandas as pd

# Directory of the raw data files
_data_root = r'../../data'

# Name of the data file
_data_file = r'dataset.csv'

# Name of the data file
_saved_model_root = r'saved_models'

# variables
text_var = "sentence"
label_var = "category"

# model threshold
threshold = 0.3


def tokenize_reviews(reviews: str) -> str:
    """
    clean accents, special characteres and stopwords(pt) on strings

    Args:
        reviews (str): any string

    Returns:
        str: cleaned string
    """    
    # nltk.download('stopwords')
    stopwords_pt = stopwords.words("portuguese")

    # lower the str to padronize the replace
    reviews = reviews.lower()

    # removing special characteres
    reviews = re.sub(r" '| '|^'|'$", " ", reviews)

    # removing accents
    reviews = re.sub(r"[àáâãäåã]", "a", reviews)
    reviews = re.sub(r"[èéêë]", "e", reviews)
    reviews = re.sub(r"[ìíîï]", "i", reviews)
    reviews = re.sub(r"[òóôõö]", "o", reviews)
    reviews = re.sub(r"[ùúûü]", "u", reviews)
    reviews = re.sub(r"[ýÿ]", "y", reviews)
    reviews = re.sub(r"[ç]", "c", reviews)

    # reducing to just letters
    reviews = re.sub(r"[^a-z\d ]", " ", reviews)

    # removing stopwords
    reviews = re.sub(r'\b(' + r'|'.join(stopwords_pt) + r')\b\s*', "", reviews)

    # removing multiple spaces
    reviews = " ".join(reviews.split())
    reviews = reviews.strip()

    return reviews


def load_and_transform_dataset(encode_target_variable: bool=False) -> pd.DataFrame:
    """
    load and transform the dataset stored on the _data_root + _data_file path.
    the features name to be cleaned have to be defined on the begin of the file.

    Returns:
        pd.DataFrame: a cleaned pandas dataframe.
    """    
    dataset = pd.read_csv(os.path.abspath(os.path.join(__file__, _data_root, _data_file)))

    # splitting the target label to get one target per sentence
    dataset[label_var] = dataset[label_var].str.split(',')
    dataset = dataset.explode(label_var).reset_index(drop=True)

    # cleaning sentences
    dataset[text_var] = dataset[text_var].apply(tokenize_reviews)
    dataset[label_var] = dataset[label_var].apply(tokenize_reviews)

    if encode_target_variable:
        # transform the categorical into numerical
        enc = LabelEncoder()
        dataset[label_var] = enc.fit_transform(dataset[label_var])

        # saving encoder to use in production
        with open(os.path.join(__file__, r"..", 'basic_implementation_label_encoder.pkl'), 'wb') as output:
            pickle.dump(enc, output)

    return dataset


def transform_predictions_to_strings(list_of_predictions: np.ndarray, threshold: int) -> List:
    """
    transform the output predictions from a keras nlp multi-classification model to strings

    Args:
        list_of_predictions (np.ndarray): a numpy 2D array containing the predictions
        threshold (int): a threshold between 0-1 to clip the predictions

    Returns:
        List: a list containing each class above threshold for the prediction
    """    
    output_list = []

    for prediction in list_of_predictions:
        # for outputs in prediction:
        above_threshold = np.where(prediction > threshold)
        list_above_threshold = above_threshold[0].tolist()
        # if none value is above threshold, get the max probability
        if len(list_above_threshold) == 0:
            string_classes = str(np.argmax(prediction))
        else:
            string_classes = ', '.join([str(num) for num in list_above_threshold])
        output_list.append(string_classes)

    return output_list


def full_label_replace(text: str, encoder: LabelEncoder) -> str:
    """
    Replace strings on the text based on the `encoder` classes 

    Args:
        text (str): the sentence to replace
        enc (LabelEncoder): the LabelEncoder object

    Returns:
        str: the sentence with the classes replaced
    """    
    encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    for key, value in encoder_dict.items():
        text = text.replace(str(value), key)

    return text


def model_builder():
    
    rate = 0.2
    kernel_initializer = "orthogonal"
    activation_function = "tanh"
    units_1 = 64
    units_2 = 32
    learning_rate = 0.001
    
    # every feature is just a string, so it is shape = (1,)
    inputs = tf.keras.Input(shape=(1,), name="sentence", dtype=tf.string)
    # flatten our tensors to 1 dimension
    reshaped_narrative = tf.reshape(inputs, [-1])
    # nn only works with numbers, so we should transform our inputs into numbers through _layerding
    # we are using universal-sentence-encoder-multilingual because our strings are in PT-BR
    embed_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    x = embed_layer(reshaped_narrative)
    # the output of the embed it is a 512 dimensional vector
    x = tf.keras.layers.Reshape((1,512), input_shape=(1,512))(x)
    # here is a feed foward neural network - ffn
    x = layers.Dense(units_1, activation=activation_function, kernel_initializer=kernel_initializer)(x)
    
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=units_1)(x, x, x)
    # dropout for regularization
    attn_output = layers.Dropout(rate)(attn_output)
    
    # add & normalization
    out1 = layers.LayerNormalization(epsilon=1e-7)(x + attn_output)
    # ffn
    ffn_output = layers.Dense(units_1, activation=activation_function, kernel_initializer=kernel_initializer)(out1)
    ffn_output = layers.Dense(units_1, kernel_initializer=kernel_initializer)(ffn_output)
    # dropout for regularization
    ffn_output = layers.Dropout(rate)(ffn_output)
    
    # add & normalization 
    x = layers.LayerNormalization(epsilon=1e-7)(out1 + ffn_output)
    # calculating the average for each patch of the feature map
    x = layers.GlobalAveragePooling1D()(x)
    # dropout for regularization
    x = layers.Dropout(rate)(x)
    # ffn
    x = layers.Dense(units_2, activation=activation_function, kernel_initializer=kernel_initializer)(x)
    # dropout for regularization
    x = layers.Dropout(rate)(x)
    # outputs in 5 classes
    outputs = layers.Dense(5, activation='softmax')(x)
    
    
    model = tf.keras.Model(inputs=inputs, outputs = outputs)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    model.summary()
    return model
