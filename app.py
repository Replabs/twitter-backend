
import os
import requests
from flask import Flask, request
from flask_cors import CORS
from time import time
from api_keys import twitter_api_key, consumer_key, consumer_secret, access_token, access_secret
from algorithm import pagerank
from db import db
from graph import initialize_graphs
from firebase_admin import auth
from models import embedding_model, sentiment_model
from firebase_admin import firestore
from pytwitter import Api
from datetime import datetime


app = Flask(__name__)

CORS(app)

app.logger.info("Starting app.")

graphs = {}

# Prepare the Graphs.
initialize_graphs(graphs)

app.logger.info("Loaded model.")

# Prepare the twitter API.
twitter_api = Api(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_secret=access_secret
)


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/proxy/oauth", methods=['POST'])
def proxy_oauth_token():
    body = request.get_json()
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    # Make the upstream api call.
    response = requests.post(
        "https://api.twitter.com/2/oauth2/token", data=body, headers=headers)

    if not response.ok:
        return response.json(), 400

    return response.json()


@app.route("/signup", methods=['POST'])
def sign_up():
    """Create a user in Firebase from a Twitter Oauth token and return a Firebase login token."""
    #
    # Use the access token to get the twitter user from the Twitter API.
    #
    access_token = request.headers['access_token']

    if not access_token:
        return {"error": "No access token."}, 400

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + access_token,
    }

    response = requests.get(
        "https://api.twitter.com/2/users/me?user.fields=profile_image_url", headers=headers)

    if not response.ok:
        return response.json(), 400

    user = response.json()['data']

    # Save the access token to the user in Firebase. This access token is used to
    # verify that the user is the correct firebase user from now on.
    user['access_token'] = access_token

    #
    # Get the user's lists.
    #
    response = twitter_api.get_user_followed_lists(
        user['id'], return_json=True)
    lists = response['data']

    #
    # Add the user to Firebase, if it doesn' zt already exist.
    #
    try:
        _ = auth.get_user(user['id'])
    except:
        auth.create_user(
            uid=user['id'],
            display_name=user['username'],
            photo_url=user['profile_image_url']
        )

    db.collection('users').document(user['id']).set(user, merge=True)

    return {"user": user, "lists": lists}


@app.route("/results", methods=['GET'])
def get_results():
    """Get the results for a user."""
    start = time()

    #
    # Verify the user's access token.
    #
    access_token = request.headers['access_token']
    user_id = request.headers['user_id']

    user = db.collection('users').document(user_id).get()
    user = user.to_dict()
    if user['access_token'] != access_token:
        return {"error": "Invalid access token."}, 400

    #
    # Get the PageRank results for the user.
    #
    lists = db.collection('lists').where(u'owner_id', u'==', user['id']).get()
    fb_user = db.collection('users').document(user['id']).get()
    list_results = {}

    for list_doc in lists:
        if list_doc.id not in graphs:
            print(f"Skipping list {list_doc.id}, not in memory.")
            continue

        # The graph of the list.
        G = graphs[list_doc.id].copy()

        if len(G.edges) == 0:
            continue

        type_results = {}

        # Add the pagerank results for each type.
        for reputation_type in fb_user.to_dict()['reputation_types']:
            # Calculate pagerank for the graph.
            pr = pagerank(
                G, None, topic_embedding=reputation_type['embedding'], n_results=4)

            # Replace the IDs of the PageRank results with the user object.
            def get_user(result):
                user = next(
                    filter(lambda y: y['id'] == result[0], list_doc.to_dict()['members']))
                user['top_tweets'] = result[2]
                return user

            # Add the usernames to the results for the type.
            type_results[reputation_type['text']] = list(map(get_user, pr))

        # Add all the type results for a list to the list results.
        list_results[list_doc.to_dict()['name']] = type_results

    #
    # Add in the default 'Geopolitics' list as a demo list.
    #
    if '1483456727219683332' not in [x.id for x in lists]:
        list_doc = db.collection(
            'lists').document('1483456727219683332').get()
        topic = 'Geopolitics'
        pr = pagerank(
            graphs[list_doc.id].copy(), topic, n_results=6)

        # Replace the IDs of the PageRank results with the user object.
        def get_geopolitics_user(result):
            user = next(
                filter(lambda y: y['id'] == result[0], list_doc.to_dict()['members']))
            user['top_tweets'] = result[2]
            return user

        list_results[list_doc.to_dict()['name']] = {
            topic: list(map(get_geopolitics_user, pr))
        }

    print(f"/results took {time() - start} seconds.")

    # The results expire in a day.
    expires_at = int(time() * 1000 + 24 * 60 * 60 * 1000)

    return {
        "lists": list_results,
        "expires_at": expires_at
    }


@app.route("/update_graphs", methods=['POST'])
def update_graphs():
    """Update the graphs in memory and on disk from the latest Firestore data.

    Parameters:
    -----
    api_key : string
        The api key for twitter.
    """

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
    G = graphs[list_id].copy()

    # Run the pagerank algorithm.
    results = pagerank(G, topic)

    # Return the results.
    return {"results": results}


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
    tweets = list(filter(lambda x: 'referenced_tweet' in x, tweets))

    # Create the embedding vectors for the text of the referenced tweet.
    embeddings = embedding_model(
        [t['referenced_tweet']['text'] for t in tweets])

    # Create the sentiment scores for the text of the tweet.
    sentiments = sentiment_model([t['text']
                                  for t in tweets])

    # Add the embeddings and sentiments to the tweets.
    for i, tweet in enumerate(tweets):
        tweet['sentiment'] = sentiments[i]
        tweet['referenced_tweet']['embedding'] = embeddings[i]

    app.logger.info(
        f"Created embeddings for {len(tweets)} tweets, took {time() - start}s.")

    #
    # Update the tweets in firestore.
    #
    batch = db.batch()

    for tweet in tweets:
        tweet['embedded_at'] = datetime.now()
        batch.update(db.collection('tweets').document(tweet['id']), tweet)

    batch.commit()

    return {"success": True}


@app.route("/onboarding_finished", methods=["POST"])
def onboarding_finished():
    """Post the lists and types to firebase, making the user ready for crawling."""

    #
    # Verify the user's access token.
    #
    access_token = request.headers['access_token']
    user_id = request.headers['user_id']

    user = db.collection('users').document(user_id).get()
    user = user.to_dict()
    if user['access_token'] != access_token:
        return {"error": "Invalid access token."}, 400

    body = request.get_json()
    reputation_type = body['type']
    reputation_lists = body['lists']

    if not reputation_type or not reputation_lists:
        return {"error": "Missing data"}, 400

    #
    # Add the selected lists to Firestore.
    #
    for list_id in reputation_lists:
        list_response = twitter_api.get_list(list_id, return_json=True)
        members_response = twitter_api.get_list_members(
            list_id, user_fields="profile_image_url", return_json=True)

        db.collection('lists').document(list_id).set({
            "id": list_id,
            "name": list_response['data']['name'],
            "owner_id": user['id'],
            "members": members_response['data']
        })

    #
    # Update the reputation type for the user in Firestore.
    #
    reputation_type_embedding = embedding_model(reputation_type)

    db.collection('users').document(user['id']).set({
        "reputation_types": [
            {
                "text": reputation_type,
                "embedding": reputation_type_embedding
            }
        ]
    }, merge=True)

    return {"success": True}


@app.route("/sync_status", methods=["GET"])
def get_sync_status():
    """Get the syncing status for a user."""
    #
    # Verify the user's access token.
    #
    access_token = request.headers['access_token']
    user_id = request.headers['user_id']

    user = db.collection('users').document(user_id).get()
    user = user.to_dict()
    if user['access_token'] != access_token:
        return {"error": "Invalid access token."}, 400

    lists = db.collection('lists').where(u'owner_id', u'==', user['id']).get()
    lists = [x.to_dict()['id'] for x in lists]

    return {
        "lists_done": len([x for x in lists if x in graphs]),
        "lists_total": len(lists)
    }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
