import pickle
import os

with open('dataset.pkl', 'rb') as f:
    temp = pickle.load(f)

print(temp.keys())
print(len(temp['metadata']))
print(len(temp['embedding_strings']))
print(temp['embeddings'].shape)


