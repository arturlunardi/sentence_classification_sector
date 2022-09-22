# About

This application provides a model that classifies the sector of sentences. The predictions can be made through an API builded with FastAPI.

The model was trained in Portuguese (PT-BR) data.

# Model

The model architecture is an adaptation of a Transformer, based on the [famous paper: Attention is All You Need.](https://arxiv.org/abs/1706.03762), builded with TensorFlow. 

The outputs of the model are probabilities that the sentence belongs to a certain class. For this project, it was needed that the sentence may have more than one class. To overcome this, it was defined a threshold that every class probability above this threshold is considered as an output.

# Run the App

To run locally, clone the repository, go to the diretory and install the requirements.

```
pip install -r requirements.txt
```

Run the Live Server:

```
uvicorn app:app
```

The docs API are available on:

```
http://127.0.0.1:8000/docs
```

Example Request/Response

```
curl -X 'POST' \
  'http://127.0.0.1:8000/sentence_classification_predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "teste",
    "Lojas de Sapato"
  ]
}'


{
  "output": [
    "educacao, varejo",
    "varejo"
  ]
}
```