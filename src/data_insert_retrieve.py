import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from deta import Deta
# from dotenv import load_dotenv


# load_dotenv(".env")
# DETA_KEY = os.getenv("DETA_KEY")

def initialize_db(DETA_KEY, DETA_BASE):
    global db, DETA_KEY_VALUE, DETA_BASE_VALUE
    DETA_KEY_VALUE, DETA_BASE_VALUE = DETA_KEY, DETA_BASE
    deta = Deta(DETA_KEY_VALUE)
    db = deta.Base(DETA_BASE_VALUE)

class StoreResults:
    def __call__(self, batch):
        deta = Deta(DETA_KEY_VALUE)
        db = deta.Base(DETA_BASE_VALUE)
        for context, embedding in zip(batch["context"], batch["embeddings"]):
            db.put({"context": context, "embedding": embedding})
        return {}

def fetch_embeddings():
    res = db.fetch()
    all_items = res.items

    while res.last:
        res = db.fetch(last=res.last)
        all_items += res.items
    
    return all_items

def semantic_search(embeds_list, query, embedding_model, k):
    embeds_data = np.empty((0,768))
    
    for item in range(len(embeds_list)):
        embeds_data = np.append(embeds_data, [embeds_list[item]['embedding']], axis=0)
    
    query_embedding = np.array(embedding_model.embed_query(query)).reshape(-1, 768)
    similar_chunks = cosine_similarity(embeds_data, query_embedding)
    most_similar_idx = np.argsort(similar_chunks.reshape(similar_chunks.shape[0],))[::-1][:k]

    relevant_chunks = [embeds_list[idx]['context'] for idx in most_similar_idx]
    semantic_contexts = ' '.join(relevant_chunks)
    return semantic_contexts
