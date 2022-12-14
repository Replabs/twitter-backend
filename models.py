import requests
from api_keys import embedding_api_key, sentiment_api_key


def embedding_model(text):
    """Run the text through the embedding model."""

    url = "https://embedding-model-bo3523uimq-uc.a.run.app/query"

    response = requests.post(url, json={
        'api_key': embedding_api_key,
        'text': text,
    })

    body = response.json()

    return body['result']


def sentiment_model(text):
    """Run the text through the sentiment model."""

    url = "https://sentiment-model-bo3523uimq-uc.a.run.app/query"

    response = requests.post(url, json={
        'api_key': sentiment_api_key,
        'text': text,
    })

    body = response.json()

    return body['result']
