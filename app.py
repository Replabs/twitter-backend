import os

from flask import Flask, request
from time import time
from sentence_transformers import SentenceTransformer
from api_keys import twitter_api_key
from db import db

app = Flask(__name__)

app.logger.info("Starting app.")

# Download the model.
embedding_model = SentenceTransformer('./model')
app.logger.info("Loaded model.")


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/twitter/query", methods=['POST'])
def query_twitter():
    """Query a twitter list, given a topic.

    Parameters:    
    -----
    api_key : string
        The api key for twitter.

    tweets : string
        The tweets to construct a graph from.

    topic : string
        The topic to query for.    
    """
    start = time()

    body = request.get_json()

    topic = body['topic']
    tweets = body['tweets']
    api_key = body['api_key']

    # Verify the API key.
    if api_key != twitter_api_key:
        app.logger.info("invalid twitter api key")
        return {"error": "invalid api key"}

    # Fetch the list members from firestore.
    ref = db.collection('lists').document(list_id).get()

    print(ref)
    print(ref.data)

    return {}

#    print(ref)

    # Create items from all firestore docs.
    #items = list(map(lambda x: {**x.to_dict(), 'id': x.id}, stream))


@app.route("/twitter/embed_all", methods=["POST"])
def embed_all():
    start = time()

    print("foo")

    # Stream the firebase docs.
    docs = db.collection('tweets').stream()

    # Create items from all firestore docs.
    tweets = map(lambda x: {**x.to_dict(), 'id': x.id}, docs)

    tweets_with_referenced_tweet = list(filter(
        lambda t: t['referenced_tweet'], tweets))

    tweets_without_referenced_tweet = list(filter(
        lambda t: not t['referenced_tweet'], tweets))

    # Create the embedding vectors from the text.
    referenced_tweet_embeddings = embedding_model.encode(
        [t['referenced_tweet']['text'] for t in tweets_with_referenced_tweet])

    # Add the embeddings to the tweets.
    for i, tweet in enumerate(tweets_with_referenced_tweet):
        tweet['sentiment'] = 1
        tweet['referenced_tweet']['embedding'] = referenced_tweet_embeddings[i].tolist()

    for tweet in tweets_without_referenced_tweet:
        tweet['sentiment'] = 1

    tweets = tweets_with_referenced_tweet + tweets_without_referenced_tweet

    app.logger.info(
        f"Created embeddings for {len(tweets)} tweets, took {time() - start}s.")

    # Update the tweets in firestore.
    for tweet in tweets:
        db.collection('tweets').document(tweet['id']).update(tweet)

    print(f"finished. Took {time() - start}")

    return {"success": True}


@app.route("/twitter/create_embeddings", methods=['POST'])
def create_twitter_embeddings():
    """Create embedding vectors for tweets. 

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

    # Update the tweets in firestore.
    for tweet in tweets:
        db.collection('tweets').document(tweet.id).update(tweet)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
