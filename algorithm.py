"""The PageRank algorithm."""

import networkx as nx
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from statistics import mean
from time import time


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


def _weigh_graph(G, topic, embedding_model, sentiment_weight=0.2, similarity_weight=1.0, topic_embedding=None):
    """Weight the graph against the topic.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The MultiDiGraph of the tweets for a list.

    topic : string
        The topic to weight the graph against.

    topic_embedding : np.array, Optional
        The embedding of the topic to weight the graph against.
        Used instead of topic if provided.

    embedding_model : SentenceTransformer
        The embedding model used to create embeddings for the topic.

    sentiment_weight : number
        How much the sentiment should affect the weights.

    similarity_weight : number
        How much the similarity should affect the weights.

    Returns : nx.DiGraph
    """

    start = time()

    # The embedding of the topic.
    if topic_embedding is None:
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
    # Combine all weights for each node-pair into one,
    # turning the multi-digraph into a digraph.
    #
    F = nx.DiGraph()

    for i, e in enumerate(list(G.edges(keys=True))):
        # For every unique node-pair, only do the summarization once.
        if not F.has_edge(*e[:2]):
            # The edge data between two nodes.
            node_pair_edges = G[e[0]][e[1]]

            # Calculate the average weight over all of the edge
            # data between two nodes.
            average_weight = mean([pow(max(0, weights[i + j]), 2)
                                  for j in range(len(node_pair_edges))])

            # Summarize the edge data.
            data = {
                'reply_tweets': [node_pair_edges[j]['reply_tweet'] for j in range(len(node_pair_edges))],
                'referenced_tweets': [node_pair_edges[j]['original_tweet'] for j in range(len(node_pair_edges))],
                'weight': average_weight
            }

            # Create an edge from the summarized edge data.
            F.add_edge(e[0], e[1], **data)

    # Return the new DiGraph.
    return F


def pagerank(G, topic, embedding_model, n_results=None, topic_embedding=None):
    """Run the main pagerank algorithm.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The graph constructed from tweets.

    topic : string
        The topic to query for.

    topic_embedding : np.array, Optional
        The embedding of the topic to query for.
        If provided, the topic is ignored.

    n_results: int, Optional
        How many nodes to include in the results.
        None means include all results.
    """

    # Weigh the graph.
    G = _weigh_graph(G, topic, embedding_model,
                     topic_embedding=topic_embedding)

    # Create a personalization vector by averaging all incoming weights and applying
    # an exponential value to the result.
    personalization = dict(
        [(n, _average_incoming_weights(G, n, 3)) for n in list(G.nodes)])

    start = time()

    # Calculate the PageRank results.
    results = nx.pagerank(G, personalization=personalization, alpha=0.55)

    print(
        f"Finished calculating pagerank for graph with {len(G.edges)} nodes. Took {time() - start} seconds.")

    # Sort the results.
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Truncate the results if applicable.
    if n_results is not None and len(results) > n_results:
        results = results[n_results]

    # Return the results.
    return results
