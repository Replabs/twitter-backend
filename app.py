import os

from flask import Flask, request
from time import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from api_keys import twitter_api_key
from algorithm import pagerank
from db import db

app = Flask(__name__)

app.logger.info("Starting app.")

# Download the embedding model.
embedding_model = SentenceTransformer('./embedding_model')

# Download the sentiment model.
sentiment_task = pipeline(
    'sentiment-analysis', model='./sentiment_model', tokenizer='./sentiment_model')

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

    list_id : string
        The ID of the list to query.

    topic : string
        The topic to query for.
    """
    print("Start query")

    start = time()

    body = request.get_json()

    api_key = body['api_key']
    list_id = body['list_id']
    topic = body['topic']

    # Verify the API key.
    if api_key != twitter_api_key:
        app.logger.info("invalid twitter api key")
        return {"error": "invalid api key"}

    # Fetch the relevant list members.
    members = db.collection('lists').document(
        list_id).get().to_dict()['members']

    # The tweets for which to run PageRank.
    tweets = []

    for member in members:
        # Get the tweets from the member.
        id = member['id']
        docs = db.collection('tweets').where(
            u'author_id', u'==', id).stream()

        # Map the documents to dictionaries.
        t = map(lambda x: {**x.to_dict(), 'id': x.id}, docs)

        #
        # Only include reference tweets that are not referencing the author,
        # and tweet that references tweets within the original list.
        #
        t = filter(
            lambda t: t['referenced_tweet']
            and t['referenced_tweet']['author_id'] != t['author_id']
            and t['referenced_tweet']['embedding'] is not None
            and t['referenced_tweet']['author_id'] in [m['id'] for m in members], t)

        # Accumulate tweets from all list members.
        tweets = tweets + list(t)

    print(
        f"Finished fetching {len(tweets)} tweets. Took {time() - start} seconds.")

    # Run the pagerank algorithm.
    results = pagerank(tweets, topic, embedding_model)

    # Return the results.
    return {"results": results}


@ app.route("/twitter/embed_all", methods=["POST"])
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


@ app.route("/twitter/create_embeddings", methods=['POST'])
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
