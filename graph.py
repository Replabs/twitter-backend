"""Functions for creating graphs of the database that are stored in memory."""

import networkx as nx
import pandas as pd
import numpy as np
from sklearn.utils import deprecated
from db import db
from os.path import exists
import pickle


def create_twitter_graph(tweets):
    """Create an nx MultiDiGraph from tweets."""
    return "foo"
