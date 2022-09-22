from fastapi import FastAPI
import pickle
from tensorflow import keras
import os
import tensorflow_text
import utils
import numpy as np
from pydantic import BaseModel
from typing import List
import model

# load model
try:
    loaded_model = keras.models.load_model(os.path.abspath(os.path.join(__file__, r"..", utils._saved_model_root, utils._model_directory)))
except:
    loaded_model = model.create_and_save_model()

# load enc
with open(os.path.abspath(os.path.join(__file__, r"..", utils.label_encoder_file)), 'rb') as loaded_enc:
    enc = pickle.load(loaded_enc)

class Sentence(BaseModel):
    sentences: List[str]

app = FastAPI()

@app.post("/sentence_classification_predict/")
async def read_sentence(sentences: Sentence):
    sentences_list = sentences.sentences
    tokenized_sentences = list(map(utils.tokenize_reviews, sentences_list))
    list_of_predictions = loaded_model.predict(tokenized_sentences)

    output_list = utils.transform_predictions_to_strings(list_of_predictions=list_of_predictions, threshold=utils.threshold)

    output_list = list(
        map(lambda x: utils.full_label_replace(x, enc), output_list)
    )

    return {"output": output_list}