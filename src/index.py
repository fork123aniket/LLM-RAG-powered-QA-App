import ray
from datasets import load_dataset

from src.embed import EmbedChunks
from src.data_insert_retrieve import StoreResults

def set_index(embedding_model_name, insert_dataset=True, contexts=None):
    if insert_dataset:
        data = load_dataset("squad", split="train[:200]")
        ray_ds = ray.data.from_huggingface(data)
    else:
        ray_ds = ray.data.from_items([
            {'context': context} for context in contexts
            ])

    # Embed chunks
    embedded_chunks = ray_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name},
        batch_size=100,
        num_gpus=1,
        concurrency=1,
    )

    # Index data
    embedded_chunks.map_batches(
        StoreResults,
        batch_size=100,
        num_cpus=1,
        concurrency=1,
    ).count()
