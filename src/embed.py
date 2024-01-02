from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding_model(embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"device": "cuda", "batch_size": 100},
    )
    return embedding_model


class EmbedChunks:
    def __init__(self, model_name):
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["context"])
        return {"context": batch["context"], "embeddings": embeddings}
