# """Script for creating a dataset for the classification model."""

# import itertools
# import json
# import random
# from time import time
# import sys
# from db import db


# from graph import initialize_graphs, twitter_graphs


# def _create_dataset(n_tweets):
#     """Create training and validation data and save it to disk."""
#     start = time()

#     tweets = []
#     members = {}

#     initialize_graphs()

#     for list_id, G in twitter_graphs.items():
#         # Add the list members to the members dictionary.
#         print(list_id)
#         l = db.collection('lists').document(list_id).get()
#         l = l.to_dict()['members'] if l.exists else []

#         for u in l:
#             members[u['id']] = u['username']

#         # Add the tweets from the list.
#         tweets.append(G.copy().edges(data=True))

#     tweets = list(itertools.chain.from_iterable(tweets))
#     print(str(len(tweets)) + " tweets in total.")
#     print(members)

#     # Shuffle the tweets.
#     seed = 1337
#     random.Random(seed).shuffle(tweets)

#     # Take the first n_tweets.
#     tweets = tweets[:n_tweets]
#     tweets = [{'is_assessment': 0, 'original': t[2]['original_tweet'], 'to': members[t[1]] if t[1] in members else '', 'reply': t[2]
#                ['reply_tweet'], 'from': members[t[0]] if t[0] in members else '', } for t in tweets]

#     # Split the tweets into training and test data.
#     train = tweets[:int(n_tweets * 0.8)]
#     test = tweets[int(n_tweets * 0.8):]

#     # Save the tweets to disk.
#     with open(f"datasets/train.json", "w") as f:
#         json.dump(train, f)

#     with open(f"datasets/test.json", "w") as f:
#         json.dump(test, f)

#     print(
#         f"Created training and validation datasets from {n_tweets} tweets. Took {time() - start} seconds.")


# def _move_tweet_collection():
#     print("Starting updating tweets")
#     docs = db.collection('tweets').select([]).get()
#     print(len(list(docs)))
#     # tweets = map(lambda x: {**x.to_dict(), 'id': x.id}, docs)

#     # for t in tweets:
#     #     db.collection('tweets_old').document(t['id']).set(t)
#     # print("Finished updating tweets")


# if __name__ == "__main__":
#     _move_tweet_collection()
#     # n_tweets = 1000

#     # if len(sys.argv) > 1:
#     #     n_tweets = int(sys.argv[1])

#     # _create_dataset(n_tweets)
