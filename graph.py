"""Initialize the graphs from firestore data."""

import os
from datetime import datetime
from os.path import exists
from db import db
import networkx as nx
import numpy as np
from time import time
from algorithm.twitter import pagerank as twitter_pagerank
from algorithm.twitter import weigh_graph as weigh_twitter_graph
from algorithm.dao import pagerank as dao_pagerank
from algorithm.dao import weigh_graph as weigh_dao_graph
from daos import daos
from models import embedding_model

# The twitter graphs, keyed by list ID.
twitter_graphs = {}

# The DAO graphs, keyed by server name.
dao_graphs = {}


def _fetch_tweets(users):
    """Fetch tweets from the database for users."""
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


def _create_twitter_edge(tweet):
    """Create a graph edge tuple from a tweet."""
    src = tweet['author_id']
    dst = tweet['referenced_tweet']['author_id']

    properties = {
        'sentiment': tweet['sentiment'] if 'sentiment' in tweet else 1,
        'embedding': np.array(tweet['referenced_tweet']['embedding']),
        'original_tweet': tweet['referenced_tweet']['text'],
        'reply_tweet_id': tweet['id'],
        'reply_tweet': tweet['text'],
    }

    return (src, dst, properties)


def _create_attestation_edge(doc):
    """Create a graph edge tuple from a DAO attestation document."""

    src = doc['from']
    dst = doc['to']

    properties = {
        'text': doc['text'],
        'embedding': np.array(doc['embedding'])
    }

    return (src, dst, properties)


def _initialize_dao_graphs():
    """
    Downloads the firestore data for each DAO and prepares a graph.
    These graphs are stored in memory and on disk for easy access.
    """
    start = time()

    day = datetime.now().strftime("%m%d%y")

    # Delete old graphs.
    for file_name in os.listdir("graphs/dao"):
        if not file_name.endswith(f"{day}.gpickle"):
            os.remove(os.path.join("graphs/dao", file_name))

    for dao in daos:
        print(f"Initializing graph {dao}.")

        # The path for the pickle file.
        path = f"graphs/dao/{dao}_{day}.gpickle"

        # If a pickled file already exists for today,
        # add it to the dictionary and continue.
        if exists(path):
            print(f"Graph {dao} already on disk, skipping...")
            G = nx.read_gpickle(path)
            dao_graphs[dao] = nx.freeze(G)
            continue

        # Get the attestations for the DAO from Firestore.
        ref = db.collection(dao)
        stream = ref.stream()
        attestations = list(map(lambda x: {**x.to_dict(), 'id': x.id}, stream))

        # Embed the attestations that lack embeddings.
        for a in attestations:
            if 'embedding' not in a:
                a['embedding'] = embedding_model(a['text'])
                ref.document(a['id']).update({'embedding': a['embedding']})

        # Create the graph.
        G = nx.MultiDiGraph()

        # Add edges from the attestations.
        G.add_edges_from(
            map(lambda x: _create_attestation_edge(x), attestations))

        # Save the graph to disk.
        nx.write_gpickle(G, path)

        # Add the graph to the dictionary.
        dao_graphs[dao] = nx.freeze(G)

        print(
            f"Created graph for {dao} with {len(attestations)} attestations.")

    print(
        f"Finished initializing {len(dao_graphs)} graphs. Took {time() - start} seconds.")


def _initialize_twitter_graphs():
    """
    Downloads the twitter data for each list and prepares a graph. 
    These graphs are stored in memory and on disk for easy access.
    """
    start = time()

    # The lists.
    lists = db.collection('lists').stream()
    lists = map(lambda x: x.to_dict(), lists)

    day = datetime.now().strftime("%m%d%y")

    # Delete old graphs.
    for file_name in os.listdir("graphs/twitter"):
        if not file_name.endswith(f"{day}.gpickle"):
            os.remove(os.path.join("graphs/twitter", file_name))

    for l in lists:
        print(f"Initializing graph {l['name']}.")

        # The path for the pickle file.
        path = f"graphs/twitter/{l['id']}_{day}.gpickle"

        # If a pickled file already exists for today,
        # add it to the dictionary and continue.
        if exists(path):
            print(f"Graph {l['id']} already on disk, skipping...")
            G = nx.read_gpickle(path)
            twitter_graphs[l['id']] = nx.freeze(G)
            continue

        # Fetch tweets for all members of the list.
        tweets = _fetch_tweets(l['members'])

        # Create the graph.
        G = nx.MultiDiGraph()

        # Add edges from the tweets.
        G.add_edges_from(map(lambda x: _create_twitter_edge(x), tweets))

        # Save the graph to disk.
        nx.write_gpickle(G, path)

        # Add the graph to the dictionary.
        twitter_graphs[l['id']] = nx.freeze(G)

        print(f"Created graph for {l['name']} with {len(tweets)} tweets.")

    print(
        f"Finished initializing {len(twitter_graphs)} graphs. Took {time() - start} seconds.")


def initialize_graphs():
    """Initialize all graphs."""
    _initialize_dao_graphs()
    _initialize_twitter_graphs()


def get_pyvis_twitter_graph(G, list_id, topic, alpha=0.55, sentiment_weight=0.2, similarity_threshold=0.0):
    """Create a graph with all of the necessary attributes for rendering in pyvis / vis.js"""

    # Weight the graph for the topic.
    G = weigh_twitter_graph(G, topic, sentiment_weight=sentiment_weight,
                            similarity_threshold=similarity_threshold)

    # Run the pagerank algorithm.
    results = twitter_pagerank(G, topic, alpha=alpha)

    # Get the list members.
    list_doc = db.collection(
        'lists').document(list_id).get()
    members = list_doc.to_dict()['members']

    print(results)

    #
    # Add the node attributes.
    #
    nx.set_node_attributes(G, values=dict(
        [(x[0], x[1] * 1000) for x in results]), name="size")
    nx.set_node_attributes(G, values=dict(
        [(x[0], next(m['username'] for m in members if m['id'] == x[0])) for x in results]), name="label")

    #
    # Add the edge attributes, clamped between 0.1 and 1.
    #
    d = {}
    for e in G.edges(data=True):
        d[(e[0], e[1])] = max(0.1, min(1, e[2]['weight'] / 100))

    nx.set_edge_attributes(G, d, name="value")

    return G


def get_pyvis_dao_graph(G, topic, alpha=0.55, similarity_threshold=0.2):
    """Create a graph with all of the necessary attributes for rendering in pyvis / vis.js"""

    # Weight the graph for the topic.
    G = weigh_dao_graph(G, topic, similarity_threshold=similarity_threshold)

    # Run the pagerank algorithm.
    results = dao_pagerank(G, topic, alpha=alpha)

    #
    # Add the node attributes.
    #
    nx.set_node_attributes(G, values=dict(
        [(x[0], max(5, min(100, x[1] * 100))) for x in results]), name="size")

    #
    # Add the edge attributes, clamped between 0.1 and 1.
    #
    d = {}
    for e in G.edges(data=True):
        d[(e[0], e[1])] = max(0.1, min(1, e[2]['weight'] / 100))

    nx.set_edge_attributes(G, d, name="value")

    return G
