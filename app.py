import os

from flask import Flask, request
from time import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from api_keys import twitter_api_key
from algorithm import pagerank
from db import db
from twitter_plugin.backend.graph import initialize_graphs

app = Flask(__name__)

app.logger.info("Starting app.")

# Download the embedding model.
embedding_model = SentenceTransformer('./embedding_model')

# Download the sentiment model.
sentiment_task = pipeline(
    'sentiment-analysis', model='./sentiment_model', tokenizer='./sentiment_model')

graphs = {}

# Prepare the Graphs.
initialize_graphs(graphs)

app.logger.info("Loaded model.")


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/update_graphs", methods=['POST'])
def update_graphs():
    """Update the graphs in memory and on disk  from the latest Firestore data."""
    start = time()

    body = request.get_json()

    # Verify the API key.
    if body['api_key'] != twitter_api_key:
        app.logger.info("invalid twitter api key")
        return {"error": "invalid api key"}

    # Initialize the graphs anew. This will take some time.
    initialize_graphs(graphs)

    print(f"Finished updating graphs. Took {time() - start} seconds.")

    return {"success": True}


@app.route("/pagerank", methods=['POST'])
def query_twitter():
    """Get the PageRank scores for a list, given a topic.

    Parameters:
    -----
    api_key : string
        The api key for twitter.

    list_id : string
        The ID of the list to query.

    topic : string
        The topic to query for.
    """
    body = request.get_json()

    list_id = body['list_id']
    topic = body['topic']

    if list_id not in graphs:
        return {"error": "Graph not initialized."}

    # The graph for the list.
    G = graphs[list_id].copy().unfreeze()

    # Run the pagerank algorithm.
    results = pagerank(G, topic, embedding_model)

    # Return the results.
    return {"results": results}


@app.route("/embed_all", methods=["POST"])
def embed_all():
    """Embed all tweets. Might take minutes to finish.

    Parameters:
    -----
    api_key : string
        The api key for twitter."""

    start = time()

    body = request.json()

    # Verify the API key.
    if body['api_key'] != twitter_api_key:
        app.logger.info("invalid twitter api key")
        return {"error": "invalid api key"}

    # Stream the firebase docs.
    docs = db.collection('tweets').stream()

    # Create items from all firestore docs.
    tweets = map(lambda x: {**x.to_dict(), 'id': x.id}, docs)

    # Exclude tweets without a referenced tweet.
    tweets = list(filter(lambda t: t['referenced_tweet'], tweets))

    # Create the embedding vectors for the text of the referenced tweet.
    embeddings = embedding_model.encode(
        [t['referenced_tweet']['text'] for t in tweets])

    # Create the sentiment scores for the text of the tweet.
    sentiments = sentiment_task([t['text']
                                for t in tweets])

    for i, tweet in enumerate(tweets):
        tweet['sentiment'] = sentiments[i]['score']
        tweet['referenced_tweet']['embedding'] = embeddings[i].tolist()

    app.logger.info(
        f"Created embeddings for {len(tweets)} tweets, took {time() - start}s.")

    # Update the tweets in firestore.
    for tweet in tweets:
        db.collection('tweets').document(tweet['id']).update(tweet)

    print(f"finished. Took {time() - start}")

    return {"success": True}


@app.route("/embed", methods=['POST'])
def create_twitter_embeddings():
    """Calculate embedding vectors and sentiment scores for tweets.

    Parameters:
    -----
    api_key : string
        The api key for twitter.

    tweets : list
        The tweets for which to calculate embeddings."""

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

    # Filter out tweets without a referenced_tweet.
    tweets = filter(lambda x: 'referenced_tweet' in x, tweets)

    # Create the embedding vectors for the text of the referenced tweet.
    embeddings = embedding_model.encode(
        [t['referenced_tweet']['text'] for t in tweets])

    # Create the sentiment scores for the text of the tweet.
    sentiments = sentiment_task([t['text']
                                for t in tweets])

    # Add the embeddings and sentiments to the tweets.
    for i, tweet in enumerate(tweets):
        tweet['sentiment'] = sentiments[i]['score']
        tweet['referenced_tweet']['embedding'] = embeddings[i].tolist()

    app.logger.info(
        f"Created embeddings for {len(tweets)} tweets, took {time() - start}s.")

    # Update the tweets in firestore.
    for tweet in tweets:
        db.collection('tweets').document(tweet.id).update(tweet)

    return {"success": True}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
