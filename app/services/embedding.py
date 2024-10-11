from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings

class EmbeddingService:
    def __init__(self):
        self.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.openai_embeddings = OpenAIEmbeddings()

    def get_sentence_transformer_embedding(self, text):
        return self.sentence_transformer_model.encode(text, convert_to_tensor=True).tolist()

    def get_openai_embedding(self, text):
        return self.openai_embeddings.embed_query(text)