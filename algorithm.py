"""The PageRank algorithm."""

import networkx as nx
import numpy as np
from scipy.spatial import distance
import time


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pagerank(G, query, embedding_model, drop_irrelevant_threshold=0.7, n_results=None):
    """Run the main pagerank algorithm.

    Parameters
    ----------
    G : nx.DiGraph

    query : string
        The search query

    drop_irrelevant_threshold : int, Optional
        Any edge similarity below this value will be removed from the graph.
        Default is 0.7.

    n_results: int, Optional
        How many nodes to include in the results. 
        None means include all results.
    """

    # Get the full edge data dictionary from an edge tuple.
    def edge_data(e): return G[e[0]][e[1]]

    query_embedding = np.expand_dims(embedding_model.encode(query), 0)
    edge_embeddings = np.asarray([edge_data(e)['embedding'] for e in G.edges])

    # Calculate the standardized similarities.
    dist = distance.cdist(query_embedding, edge_embeddings, "cosine")[0]
    similarities = 1 - _sigmoid((dist - dist.mean()) / dist.std())

    for i, e in enumerate(list(G.edges)):
        edge_data(e)['weight'] = similarities[i]

    # Remove irrelevant edges.
    G.remove_edges_from(
        [e for e in G.edges if edge_data(e)['weight'] < drop_irrelevant_threshold])

    # Run the algorithm.
    results = nx.pagerank(G)

    #
    # Sort based on pagerank score, and drop irrelevant results.
    #
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    if n_results:
        results = results[:n_results]

    #
    # Format each result to include name, score and a short excerpt.
    #
    def format_result(r):
        info = {
            'name': r[0],
            'score': r[1],
            'excerpt': [],
            'in_edges': len(G.in_edges(r[0])),
            'out_edges': len(G.out_edges(r[0])),
        }

        # Append the top 3 matching reasons.
        for e in sorted(list(G.in_edges(r[0])), key=lambda x: edge_data(x)['weight'], reverse=True)[:3]:
            if 'text' in edge_data(e):
                info['excerpt'].append(edge_data(e)['text'])
            elif 'reason' in edge_data(e):
                info['excerpt'].append(edge_data(e)['reason'])

        return info

    # Return the formatted results.
    return {'results': [format_result(r) for r in results]}
