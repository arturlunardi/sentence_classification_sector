from fastapi import FastAPI
import pickle
from tensorflow import keras
import os
import tensorflow_text
import utils
import numpy as np
from pydantic import BaseModel
from typing import List

# load model
model = keras.models.load_model(os.path.join(utils._saved_model_root, 'basic_serving_model'))

# load enc
with open('basic_implementation_label_encoder.pkl', 'rb') as loaded_enc:
    enc = pickle.load(loaded_enc)

class Sentence(BaseModel):
    sentences: List[str]

app = FastAPI()

@app.post("/basic_implementation/")
async def read_sentence(sentences: Sentence):
    sentences_list = sentences.sentences
    list_of_predictions = model.predict(sentences_list)

    output_list = utils.transform_predictions_to_strings(list_of_predictions=list_of_predictions, threshold=utils.threshold)

    output_list = list(
        map(lambda x: utils.full_label_replace(x, enc), output_list)
    )

    return {"output": output_list}