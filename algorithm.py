"""The PageRank algorithm."""

import networkx as nx
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from statistics import mean
from time import time


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


def _create_graph(tweets, topic, embedding_model, sentiment_weight=0.2, similarity_weight=1.0):
    """Create weights for the graph from sentiment and query similarity"""

    start = time()

    # The multi Di-Graph.
    G = nx.MultiDiGraph()

    # Add edges to the graph.
    G.add_edges_from(map(lambda x: _create_edge(x), tweets,))

    print(
        f'Created MultiDiGraph with {len(G.edges)} edges and {len(G.nodes)} nodes.')

    # The embedding of the topic.
    topic_embedding = np.expand_dims(embedding_model.encode(topic), 0)

    # The embeddings of each edge in the MultiDiGraph.
    edge_embeddings = np.asarray(
        [np.array(G[e[0]][e[1]][e[2]]['embedding']) for e in G.edges(keys=True)])

    # Calculate the standardized similarities.
    distances = distance.cdist(topic_embedding, edge_embeddings, "cosine")[0]
    similarities = minmax_scale(1 - distances, feature_range=(0, 1))

    # Calculate the standardized sentiments.
    sentiments = np.asarray(
        [G[e[0]][e[1]][e[2]]['sentiment'] for e in G.edges(keys=True)])
    sentiments = minmax_scale(sentiments, feature_range=(-1, 1))

    # Calculate the weights.
    weights = np.asarray([sen * sentiment_weight + sim * similarity_weight for sen,
                         sim in zip(sentiments, similarities)])

    #
    # Combine all weights for each user pair into one,
    # turning the multi-digraph into a digraph.
    #
    F = nx.DiGraph()

    for i, e in enumerate(list(G.edges(keys=True))):
        # For every unique node-pair, only do the summarization once.
        if not F.has_edge(*e[:2]):
            # The edge data between two nodes.
            edges_between_nodepair = G[e[0]][e[1]]

            # Calculate the average weight over all of the edge
            # data between two nodes.
            average_weight = mean([pow(max(0, weights[i + j]), 2)
                                  for j in range(len(edges_between_nodepair))])

            # Summarize the edge data.
            data = {
                'reply_tweets': [edges_between_nodepair[j]['reply_tweet'] for j in range(len(edges_between_nodepair))],
                'referenced_tweets': [edges_between_nodepair[j]['original_tweet'] for j in range(len(edges_between_nodepair))],
                'weight': average_weight
            }

            # Create an edge from the summarized edge data.
            F.add_edge(e[0], e[1], **data)

    print(
        f"Finished creating graph for {len(tweets)} tweets. Took {time() - start} seconds.")

    return F


def _edge_data(G, e):
    """Return the data dictionary from an edge tuple."""
    return G[e[0]][e[1]]


def _average_incoming_weights(G, node, exp=None):
    """Average the incoming weights of a node.

    Parameters
    ----------
    G : nx.DiGraph

    node : string
        The node key.

    exponent : int, optional
        If not None, return the exponential of the sum.
    """

    if len(G.in_edges(node)) == 0:
        return 0

    # Calculate the mean weight for the Graph.
    mean_incoming = mean([_edge_data(G, e)['weight']
                         for e in G.in_edges(node)])

    if exp is not None:
        return pow(mean_incoming, exp)

    return mean_incoming


def pagerank(tweets, topic, embedding_model, n_results=None):
    """Run the main pagerank algorithm.

    Parameters
    ----------
    tweets : list
        A list of tweets.

    topic : string
        The topic to query for.

    n_results: int, Optional
        How many nodes to include in the results.
        None means include all results.
    """

    # The graph.
    G = _create_graph(tweets, topic, embedding_model)

    # Create a personalization vector by averaging all incoming weights and applying
    # an exponential value to the result.
    personalization = dict(
        [(n, _average_incoming_weights(G, n, 3)) for n in list(G.nodes)])

    start = time()

    # Calculate the PageRank results.
    results = nx.pagerank(G, personalization=personalization, alpha=0.55)

    print(
        f"Finished calculating pagerank for graph with {len(G.edges)} nodes. Took {time() - start} seconds.")

    # Return the sorted results.
    return sorted(results.items(), key=lambda x: x[1], reverse=True)
