"""The PageRank algorithm for twitter graphs."""

import networkx as nx
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from statistics import mean
from time import time
import itertools

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


def sentiment_score(sentiment):
    """Get the sentiment score from the sentiment"""

    # Legacy sentiment scores are floats, not dicts.
    if type(sentiment) is float:
        return sentiment

    if sentiment['label'] == 'Negative':
        return -1 * sentiment['score']
    elif sentiment['label'] == 'Positive':
        return 1 * sentiment['score']

    return 0


def weigh_graph(G, topic, sentiment_weight=0.2, similarity_threshold=0.0, topic_embedding=None):
    """Weight the graph against the topic, converting the MultiDiGraph into a DiGraph.

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
    # Calculate the standardized sentiments.
    #
    sentiments = np.asarray(
        [sentiment_score(G[e[0]][e[1]][e[2]]['sentiment']) for e in G.edges(keys=True)])
    sentiments = minmax_scale(sentiments, feature_range=(-1, 1))

    # Calculate the weights.
    weights = np.asarray([sen * sentiment_weight + sim for sen,
                         sim in zip(sentiments, similarities)])

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
            average_weight = mean([pow(max(0, weights[i + j]), 2)
                                  for j in range(len(node_pair_edges))])

            # The indices in the node-pair edges.
            indices = [idx for idx in list(node_pair_edges)]

            #
            # Summarize tweet info for each tweet in the node pair, sorted on weight.
            #
            def tweet_info(node_pair, tweets):
                idx = [i for i, d in enumerate(
                    tweets) if node_pair['reply_tweet_id'] == d['reply_tweet_id']][0] + i

                return {
                    'reply_tweet_id': node_pair['reply_tweet_id'],
                    'similarity': similarities.tolist()[idx],
                    'sentiment': node_pair['sentiment']['label'],
                    'weight': weights[idx],
                    'url': f"https://twitter.com/{e[0]}/status/{node_pair['reply_tweet_id']}",
                }

            tweets = [node_pair_edges[idx] for idx in indices]
            tweets = [tweet_info(x, tweets) for x in tweets]
            tweets = sorted(tweets, key=lambda x: x['weight'], reverse=True)

            #
            # Summarize the data for the node pair.
            #
            data = {
                # The tweets for the node pair.
                'tweets': tweets,
                # The summarized weight for the node pair.
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
    if G.is_multigraph():
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

    # Add the top tweets to the results.
    def add_top_tweets(result):
        user_edges = filter(lambda e: e[1] == result[0], G.in_edges)

        highest_weights = list(sorted(
            user_edges, key=lambda e: G[e[0]][e[1]]['weight'], reverse=True))[:3]

        #
        # Get a flattened list of the tweet urls for each node pair.
        #
        top_tweet_urls = [[t['url'] for t in G[e[0]][e[1]]['tweets']]
                          for e in highest_weights]
        top_tweet_urls = list(itertools.chain(*top_tweet_urls))

        return (result[0], result[1], top_tweet_urls)

    # Return the results.
    return [add_top_tweets(r) for r in results]
