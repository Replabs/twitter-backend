import os

from flask import Flask, request
from collections import Counter
from time import time
from sentence_transformers import SentenceTransformer
from graph import get_algovera_graph, get_tec_graph
from algorithm import pagerank
from api_keys import tec, algovera, twitter_api_key
from db import db

app = Flask(__name__)

app.logger.info("Starting app.")

# Download the model.
embedding_model = SentenceTransformer('./model')
app.logger.info("Loaded model.")

# Load the tec graph from the data.
G_tec = get_tec_graph()
app.logger.info("Loaded TEC graph.")

# Load the algovera graph from the data.
G_algovera = get_algovera_graph(embedding_model)
app.logger.info("Loaded algovera graph.")


def _verify_api_key(server, key):
    """Verify the api key for the server"""

    invalid_algovera = server == "algovera" and key != algovera
    invalid_tec = server == "tec" and key != tec
    invalid_server = server != "algovera" and server != "tec"

    if invalid_algovera or invalid_tec:
        raise Exception("invalid api key")
    elif invalid_server:
        raise Exception("invalid server")


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/create_embedding", methods=['POST'])
def create_embedding():
    """Create an embedding vector for a firestore entry. 
    Meant to be called from a firebase cloud function hook.

    Parameters:    
    -----
    server : string
        The server to which the entry was added.

    api_key : string
        The api key for the server.

    doc_id : string
        The Firestore ID of the added entry.

    text : string
        The textual content of the added entry."""

    body = request.get_json()

    api_key = body['api_key']
    server = body['server']
    doc_id = body['id']
    text = body['text']

    # Verify the API key.
    try:
        _verify_api_key(server, api_key)
    except Exception as e:
        app.logger.info(str(e))
        return {"message": str(e)}

    # Create the embedding vector from the text.
    embedding = embedding_model.encode(text)

    # Add the embedding to the document in Firestore.
    db.collection(server).document(doc_id).update(
        {'embedding': embedding.tolist()})

    return {"message": "success!"}


@app.route("/query", methods=['POST'])
def query():
    """Query the PageRank model for the specified server.

    Parameters:    
    -----
    server : string
        The server to query for.

    api_key : string
        The api key for the server.

    query : string
        The textual query."""

    body = request.get_json()
    query = body['query']
    server = body['server']
    api_key = body['api_key']

    # Verify the API key.
    try:
        _verify_api_key(server, api_key)
    except:
        return {"message": "invalid api key"}

    start = time()

    # Run the pagerank model.
    if server == "algovera":
        results = pagerank(G_algovera.copy(), query,
                           embedding_model, drop_irrelevant_threshold=0.0, n_results=3)
    elif server == "tec":
        results = pagerank(G_tec.copy(), query, embedding_model, n_results=3)

    app.logger.info("PageRank model for {} took {} seconds".format(
        server, time() - start))

    # Return the results.
    return results


@app.route("/twitter/create_embeddings", methods=['POST'])
def create_twitter_embeddings():
    """Create embedding vectors for tweets. 

    Parameters:    
    -----
    api_key : string
        The api key for twitter.

    text : string
        The textual content of the added entry."""

    start = time()

    body = request.get_json()

    # The API key.
    api_key = body['api_key']

    # The tweets.
    tweets = body['tweets']

    # Verify the API key.
    if api_key != twitter_api_key:
        app.logger.info("invalid twitter api key")
        return {"error": "invalid api key"}

    app.logger.info(f"Creating embeddings for {len(tweets)} tweets...")

    # Create the embedding vectors from the text.
    tweet_embeddings = embedding_model.encode(
        [t['text'] for t in tweets])
    referenced_tweet_embeddings = embedding_model.encode(
        [t['referenced_tweet']['text'] for t in tweets])

    # Add the embeddings to the tweets.
    for i, tweet in enumerate(tweets):
        tweet['embedding'] = tweet_embeddings[i].tolist()
        tweet['referenced_tweet']['embedding'] = referenced_tweet_embeddings[i].tolist()

    app.logger.info(
        f"Created embeddings for {len(tweets)} tweets, took {time() - start}s.")

    # Return the updated tweets.
    return {"tweets": tweets}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
