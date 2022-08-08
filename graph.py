"""Initialize the graphs from firestore data."""

import datetime
from os.path import exists
from db import db
import networkx as nx
import numpy as np
from time import time


def _fetch_tweets(users):
    tweets = []

    for user in users:
        # Get the tweets from the member.
        id = user['id']
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
            and t['referenced_tweet']['author_id'] in [u['id'] for u in users], t)

        # Accumulate tweets from all list members.
        tweets = tweets + list(t)

    return tweets


def _create_edge(tweet):
    src = tweet['author_id']
    dst = tweet['referenced_tweet']['author_id']

    properties = {
        'sentiment': tweet['sentiment'] if 'sentiment' in tweet else 1,
        'embedding': np.array(tweet['referenced_tweet']['embedding']),
        'original_tweet': tweet['referenced_tweet']['text'],
        'reply_tweet': tweet['text'],
    }

    print(properties['embedding'].shape)

    return (src, dst, properties)


def initialize_graphs(graphs):
    """
    Downloads the twitter data for each list and prepares a graph. 
    These graphs are stored in memory and on disk for easy access.
    """
    start = time()

    # The lists.
    lists = db.collection('lists').stream()
    lists = map(lambda x: x.to_dict(), lists)

    for l in lists:
        print(f"Initializing graph {l['name']}.")
        name = l['name']
        day = datetime.now().strftime("%m%d%y")

        # The path for the pickle file.
        path = f"graphs/{name}_{day}.gpickle"

        # If a pickled file already exists for today, return it.
        if exists(path):
            G = nx.read_gpickle(path)
            return nx.freeze(G)

        # Fetch tweets for all members of the list.
        tweets = _fetch_tweets(l['members'])

        # Create the graph.
        G = nx.MultiDiGraph()

        # Add edges from the tweets.
        G.add_edges_from(map(lambda x: _create_edge(x), tweets,))

        #
        # Save the graph to disk.
        #
        name = l['name']
        day = datetime.now().strftime("%m%d%y")
        nx.write_gpickle(G, f"graphs/{name}_{day}.gpickle")

        # Add the graph to the dictionary.
        graphs[l['id']] = nx.freeze(G)

    print(
        f"Finished initializing {len(list(lists))} graphs. Took {time() - start} seconds.")
