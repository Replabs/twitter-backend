"""The PageRank algorithm for DAO graphs."""

import networkx as nx
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from statistics import mean
from time import time

from models import embedding_model


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


def weigh_graph(G, topic, similarity_threshold=0.0, topic_embedding=None):
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

    sentiment_weight : number
        How much the sentiment should affect the weights.

    similarity_threshold : number
        Any similarity score below the threshold will be ignored.

    Returns : nx.DiGraph
    """

    start = time()

    print(
        f"Weighing graph with {len(G.edges)} edges and {len(G.nodes)} nodes.")

    # The embedding of the topic.
    if topic_embedding is None:
        topic_embedding = embedding_model(topic)

    # The embeddings of each edge in the MultiDiGraph.
    edge_embeddings = np.asarray(
        [np.array(G[e[0]][e[1]][e[2]]['embedding']) for e in G.edges(keys=True)])

    # Calculate the standardized similarities.
    distances = distance.cdist(np.expand_dims(
        topic_embedding, 0), edge_embeddings, "cosine")[0]
    similarities = minmax_scale(1 - distances, feature_range=(0, 1))

    #
    # Remove all edges below the similarity threshold.
    #
    edges_to_remove = []
    tmp = []

    for (edge, similarity) in zip(G.edges(keys=True), similarities):
        if similarity < similarity_threshold:
            edges_to_remove.append(edge)
        else:
            tmp.append(similarity)

    G.remove_edges_from(edges_to_remove)
    similarities = np.asarray(tmp)

    #
    # Combine all weights for each node-pair into one,
    # turning the multi-digraph into a digraph.
    #

    D = nx.DiGraph()

    for i, e in enumerate(list(G.edges(keys=True))):
        # For every unique node-pair, only do the summarization once.
        if not D.has_edge(*e[:2]):
            # The edge data between two nodes.
            node_pair_edges = G[e[0]][e[1]]

            # Calculate the average weight over all of the edge
            # data between two nodes.
            average_weight = mean([pow(max(0, similarities[i + j]), 2)
                                  for j in range(len(node_pair_edges))])

            # The indices in the node-pair edges.
            indices = [idx for idx in list(node_pair_edges)]

            #
            # Summarize assessment info for each assessment in the node pair, sorted on weight.
            #
            def assessment_info(node_pair, tweets):
                return {
                    'text': node_pair['text'],
                    'similarity': similarities.tolist()[i + tweets.index(node_pair)],
                }

            #
            # The IDs of the reply tweets, sorted on weight.
            #
            assessments = [node_pair_edges[idx] for idx in indices]
            assessments = [assessment_info(x, assessments)
                           for x in assessments]
            assessments = sorted(
                assessments, key=lambda x: x['similarity'], reverse=True)

        # Summarize the edge data.
        data = {
            'assessments': assessments,
            'weight': average_weight,
        }

        # Create an edge from the summarized edge data.
        D.add_edge(e[0], e[1], **data)

    # Return the new DiGraph.
    return D


def pagerank(G, topic, n_results=None, topic_embedding=None, alpha=0.55):
    """Run the main pagerank algorithm.

    Parameters
    ----------
    G : nx.MultiDiGraph | nx.DiGraph
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

    # Weigh the graph if needed.
    if G is nx.MultiDiGraph:
        G = weigh_graph(G, topic, topic_embedding=topic_embedding)

    # Create a personalization vector by averaging all incoming weights and applying
    # an exponential value to the result.
    personalization = dict(
        [(n, _average_incoming_weights(G, n, 3)) for n in list(G.nodes)])

    start = time()

    # Calculate the PageRank results.
    results = nx.pagerank(G, personalization=personalization, alpha=alpha)

    print(
        f"Finished calculating pagerank for graph with {len(G.edges)} nodes. Took {time() - start} seconds.")

    # Sort the results.
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Truncate the results if applicable.
    if n_results is not None and len(results) > n_results:
        results = results[: n_results]

    # Return the results.
    return results
