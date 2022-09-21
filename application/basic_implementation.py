#%%
import tensorflow as tf
from tensorflow.keras import layers  
import tensorflow_hub as hub
import tensorflow_text
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
# %%
# Directory of the raw data files
_data_root = r'../data'

# Name of the data file
_data_file = r'dataset.csv'
# %%
df = pd.read_csv(os.path.join(_data_root, _data_file))
# %%
df
# %%
# splitting the target label to get one target per sentence
df['category'] = df['category'].str.split(',')
df = df.explode("category").reset_index(drop=True)
# %%
# check if there are null values
df.isnull().sum()
#%%
# check unique target values
df["category"].value_counts()
#%%
nltk.download('stopwords')
stopwords = stopwords.words("portuguese")
# %%
def tokenize_reviews(reviews):
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
    reviews = re.sub("[^a-z ]", " ", reviews)

    # removing stopwords
    reviews = re.sub(r'\b(' + r'|'.join(stopwords) + r')\b\s*', "", reviews)

    # removing multiple spaces
    reviews = " ".join(reviews.split())
    reviews = reviews.strip()

    return reviews
# %%
df["sentence"] = df["sentence"].apply(tokenize_reviews)
df["category"] = df["category"].apply(tokenize_reviews)
# %%
df["category"].value_counts()
#%%
map_category = {
    "orgao publico": 0,
    "educacao": 1,
    "industrias": 2,
    "varejo": 3,
    "financas": 4
}

df["category"] = df["category"].map(map_category)
#%%
df["category"].value_counts()
# %%
embed_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
#%%
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
#%%
from sklearn.model_selection import train_test_split

x = df["sentence"]
y = df["category"]

# split for train and test
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=0)
# %%
model = model_builder()

model.fit(x_train,
        y_train,
        validation_data = (x_test, y_test),
        steps_per_epoch = 1000, 
        validation_steps= 1000,
)
#%%
df_evaluate = df.iloc[x_test.index]
#%%
list_of_predictions = model.predict(df_evaluate["sentence"].values)
df_evaluate["y_predict"] = np.argmax(list_of_predictions, axis=1)

map_inverse_category = {v: k for k, v in map_category.items()}

df_evaluate["category"] = df_evaluate["category"].map(map_inverse_category)
df_evaluate["y_predict"] = df_evaluate["y_predict"].map(map_inverse_category)
# %%
df_evaluate
# %%
cm = confusion_matrix(df_evaluate['category'], df_evaluate['y_predict'], labels=["orgao publico", "educacao", "industrias", "varejo", "financas"], normalize=None)
cm_true = confusion_matrix(df_evaluate['category'], df_evaluate['y_predict'], labels=["orgao publico", "educacao", "industrias", "varejo", "financas"], normalize='true')
cm_columns = confusion_matrix(df_evaluate['category'], df_evaluate['y_predict'], labels=["orgao publico", "educacao", "industrias", "varejo", "financas"], normalize='pred')
#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["orgao publico", "educacao", "industrias", "varejo", "financas"])

disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true,
                              display_labels=["orgao publico", "educacao", "industrias", "varejo", "financas"])

disp_columns = ConfusionMatrixDisplay(confusion_matrix=cm_columns,
                              display_labels=["orgao publico", "educacao", "industrias", "varejo", "financas"])

disp.plot(ax=ax1)
disp_true.plot(ax=ax2)
disp_columns.plot(ax=ax3)

ax1.set_title('Total')
ax2.set_title('% Rows')
ax3.set_title('% Columns')

plt.tight_layout
plt.show()
# %%
precision, recall, fscore, support = precision_recall_fscore_support(df_evaluate['category'], df_evaluate['y_predict'], average='macro')
metric_accuracy_score = accuracy_score(df_evaluate['category'], df_evaluate['y_predict'])
metric_balanced_accuracy_score = balanced_accuracy_score(df_evaluate['category'], df_evaluate['y_predict'])
print(f"Precision: {precision}, Recall: {recall}, FScore: {fscore}, Acc_score: {metric_accuracy_score}, Balanced Acc_score: {metric_balanced_accuracy_score}")
# %%
