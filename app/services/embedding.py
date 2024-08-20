from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed_text(self, text):
        return self.model.embed_query(text)