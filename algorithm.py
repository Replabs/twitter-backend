"""The PageRank algorithm."""
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


def _weigh_graph(G, topic, sentiment_weight=0.2, similarity_weight=1.0, topic_embedding=None):
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

    similarity_weight : number
        How much the similarity should affect the weights.

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

    # Calculate the standardized sentiments.
    sentiments = np.asarray(
        [sentiment_score(G[e[0]][e[1]][e[2]]['sentiment']) for e in G.edges(keys=True)])
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

            reply_tweet_ids = [node_pair_edges[j]['reply_tweet_id']
                               for j in range(len(node_pair_edges))]
            reply_tweet_ids = sorted(
                reply_tweet_ids, key=lambda x: weights[i + reply_tweet_ids.index(x)], reverse=True)

            referenced_tweets = [node_pair_edges[j]['original_tweet']
                                 for j in range(len(node_pair_edges))]
            referenced_tweets = sorted(
                referenced_tweets, key=lambda x: weights[i + referenced_tweets.index(x)], reverse=True)

            # Summarize the edge data.
            data = {
                'reply_tweet_ids': reply_tweet_ids,
                'referenced_tweets': referenced_tweets,
                'weight': average_weight
            }

            # Create an edge from the summarized edge data.
            F.add_edge(e[0], e[1], **data)

    # Return the new DiGraph.
    return F


def pagerank(G, topic, n_results=None, topic_embedding=None):
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
    G = _weigh_graph(G, topic, topic_embedding=topic_embedding)

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
        results = results[: n_results]

    # Add the top tweets to the results.
    def add_top_tweets(result):
        user_edges = filter(lambda e: e[1] == result[0], G.in_edges)

        highest_weights = list(sorted(
            user_edges, key=lambda e: G[e[0]][e[1]]['weight'], reverse=True))[:3]

        def twitter_url(edge):
            return f"https://twitter.com/{edge[0]}/status/{G[edge[0]][edge[1]]['reply_tweet_ids'][0]}"

        top_tweets = [twitter_url(e) for e in highest_weights]

        return (result[0], result[1], top_tweets)

    # Return the results.
    return [add_top_tweets(r) for r in results]
