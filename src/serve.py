from fastapi import FastAPI
from pydantic import BaseModel
import os
from ray import serve

from src.qa_agent import QueryAgent
from src.index import set_index
from src.data_insert_retrieve import initialize_db


app = FastAPI()


class Query(BaseModel):
    query: str


class Answer(BaseModel):
    answer: str


@serve.deployment(
    route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 6, "num_gpus": 1}
)
@serve.ingress(app)
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

    def predict(self, query):

        answer = self.llm_qa_agent(query, self.num_chunks)

        return {'answer': answer}
    
    @app.post("/query")
    def query(self, query: Query) -> Answer:
        result = self.predict(query)
        return Answer.parse_obj(result)
    

# Deploy the Ray Serve app
deployment = Ray_LLM_QA.bind(
    data_key=os.environ["DETA_KEY"],
    deta_base=os.environ["DETA_BASE"],
    embedding_model_name="thenlper/gte-large",
    num_chunks=1,
    is_insert_data=False,
    context=None,
    no_save=True,
    path=os.environ["MODEL_PATH"],
    config_path=os.environ["CONFIG_PATH"],
)
serve.run(deployment)
