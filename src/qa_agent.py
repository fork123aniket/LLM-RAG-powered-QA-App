from src.index import set_index
from src.embed import get_embedding_model
from src.generate_answer import generate_response
from src.data_insert_retrieve import fetch_embeddings, semantic_search, initialize_db


class QueryAgent:
    def __init__(
        self,
        embedding_model_name,
        path=None, config_path=None,
    ):

        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name
        )
        self.llm = generate_response(path, config_path)

    def __call__(
        self,
        query,
        num_chunks=5,
    ):

        # Fetching all embeddings
        embeddings = fetch_embeddings()

        # Get top_k context
        context_results = semantic_search(
            embeds_list=embeddings, query=query,
            embedding_model=self.embedding_model, k=num_chunks
        )

        answer = self.llm(query=query, context=context_results)

        return answer
    

class Ray_LLM_QA:
    def __init__(
        self, data_key, deta_base, embedding_model_name, num_chunks,
        is_insert_data=True, context=None, no_save=False, path=None, config_path=None):

        initialize_db(data_key, deta_base)

        # Embedding Model
        self.embedding_model_name = embedding_model_name

        self.num_chunks = num_chunks

        if not no_save:
            set_index(self.embedding_model_name, is_insert_data, context)

        self.llm_qa_agent = QueryAgent(self.embedding_model_name, path, config_path)

    def __call__(self, query):

        answer = self.llm_qa_agent(query, self.num_chunks)

        return answer
