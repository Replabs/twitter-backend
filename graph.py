"""Functions for creating graphs of the database that are stored in memory."""

import networkx as nx
import pandas as pd
import numpy as np
from sklearn.utils import deprecated
from db import db
from os.path import exists
import pickle


@deprecated
def _load_tec_data_from_csv():
    """Returns a Pandas DataFrame from the TEC praise csv sheet."""

    df = pd.read_csv("./data/tec_discord_bot_sheet.csv")

    # Drop unnecessary columns.
    df.drop(columns=list(df.columns)[6:], inplace=True)

    # Drop first 70 rows (contains testing data).
    df.drop(df.index[0:70], inplace=True)

    # Drop rows without valid data.
    df.drop(df[df['REASON'] != df['REASON']].index, inplace=True)

    return df


def _load_algovera_data(embedding_model):
    """Returns a pandas DataFrame from the algovera data in Firebase.
    If a document misses an embedding, create one and update Firestore."""

    # Stream the algovera docs.
    stream = db.collection('algovera').stream()

    # Create items from all firestore docs.
    items = list(map(lambda x: {**x.to_dict(), 'id': x.id}, stream))

    # Add missing embeddings.
    for i in items:
        if 'embedding' not in i:
            i['embedding'] = embedding_model.encode(i['text']).tolist()

            db.collection('algovera').document(i['id']).update({
                'embedding': i['embedding']
            })

    # Create a pandas DataFrame from the items.
    df = pd.DataFrame(items)

    df.set_index('id', inplace=True)

    return df


def _load_tec_data():
    """Returns a pandas DataFrame of the tec praise data in Firestore."""

    # Stream the firebase docs.
    docs = db.collection('tec').stream()

    # Create items from all firestore docs.
    items = list(map(lambda x: {**x.to_dict(), 'id': x.id}, docs))

    # Create a pandas DataFrame from the items.
    df = pd.DataFrame(items)

    df.set_index('id', inplace=True)

    return df


def _create_tec_edge(doc):
    """Creates an edge in the graph from a Firestore document."""

    src = doc['from']
    dst = doc['to']

    properties = {
        'reason': doc['reason'],
        'date': doc['date'],
        'server': doc['server'],
        'channel': doc['channel'],
        'embedding': np.array(doc['embedding'])
    }

    return (src, dst, properties)


def _create_algovera_edge(doc):
    """Creates an edge in the graph from a Firestore document."""

    src = doc['from']
    dst = doc['to']

    properties = {
        'text': doc['text'],
        'embedding': np.array(doc['embedding'])
    }

    return (src, dst, properties)


def _create_graph(df, server, min_in_edges=None, min_out_edges=None):
    """Creates an nx DiGraph with weighted edges.

        min_in_edges : int, Optional
            How many in_edges are required per node in order for it to be included.

        min_out_edges : int, Optional
            How many out_edges are required per node in order for it to be included.
    """

    def _create_edge(doc):
        if server == "tec":
            return _create_tec_edge(doc)
        elif server == "algovera":
            return _create_algovera_edge(doc)
        else:
            raise Exception("invalid server")

    G = nx.DiGraph()
    G.add_edges_from([_create_edge(r[1]) for r in df.iterrows()])

    #
    # Filter only for nodes with both an in and an outconnection if necessary.
    # max_out_edges and min_out_edges can be adjusted depending on graph sparsity.
    #
    to_remove = []

    for n in G.nodes:
        if min_in_edges and len(G.in_edges(n)) < min_in_edges:
            to_remove.append(n)
        elif min_out_edges and len(G.out_edges(n)) < min_out_edges:
            to_remove.append(n)

    G.remove_nodes_from(to_remove)

    return G


def get_algovera_graph(embedding_model):
    """Create an nx DiGraph from firebase data."""

    path = 'graphs/algovera.gpickle'

    # Return the pickled graph if it exists.
    if exists(path):
        G = nx.read_gpickle(path)
        return nx.freeze(G)

    # Create the graph if it doesn't exist.
    data = _load_algovera_data(embedding_model)
    G = _create_graph(data, "algovera")
    nx.write_gpickle(G, path)

    return nx.freeze(G)


def get_tec_graph():
    """Create an nx DiGraph from firebase data."""

    path = 'graphs/tec.gpickle'

    # Return the pickled graph if it exists.
    if exists(path):
        G = nx.read_gpickle(path)
        return nx.freeze(G)

    # Create the graph if it doesn't exist.
    data = _load_tec_data()
    G = _create_graph(data, "tec", min_in_edges=3)
    nx.write_gpickle(G, path)

    return nx.freeze(G)
