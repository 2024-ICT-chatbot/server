import os
import logging
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re
import hashlib

class VectorStore:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings()
        self.general_vector_store = None
        self.law_vector_store = None
        self.general_faiss = None
        self.law_faiss = None
        self.general_documents = []
        self.law_documents = []
        self.document_hashes = set()
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def hash_document(self, doc):
        cleaned_content = self.clean_text(doc.page_content)
        return hashlib.md5(cleaned_content.encode()).hexdigest()

    def add_documents(self, documents, is_law_related=False):
        target_documents = self.law_documents if is_law_related else self.general_documents
        unique_documents = []

        for doc in documents:
            doc_hash = self.hash_document(doc)
            if doc_hash not in self.document_hashes:
                doc.page_content = self.clean_text(doc.page_content)
                unique_documents.append(doc)
                self.document_hashes.add(doc_hash)

        if unique_documents:
            target_documents.extend(unique_documents)
            self.logger.info(f"Added {len(unique_documents)} unique documents to the {'law' if is_law_related else 'general'} vector store.")
            self.create_vector_store(is_law_related)

    def create_vector_store(self, is_law_related=False):
        target_documents = self.law_documents if is_law_related else self.general_documents

        chroma_store = Chroma.from_documents(documents=target_documents, embedding=self.embedding_model)
        faiss_store = FAISS.from_documents(documents=target_documents, embedding=self.embedding_model)

        if is_law_related:
            self.law_vector_store = chroma_store
            self.law_faiss = faiss_store
        else:
            self.general_vector_store = chroma_store
            self.general_faiss = faiss_store

        self.logger.info(f"{'Law' if is_law_related else 'General'} vector stores created with {len(target_documents)} documents.")

    def similarity_search(self, query, is_law_related=False, k=8):
        chroma_store = self.law_vector_store if is_law_related else self.general_vector_store
        faiss_store = self.law_faiss if is_law_related else self.general_faiss

        if chroma_store is None or faiss_store is None:
            raise ValueError("Vector stores are not initialized")

        # Chroma 검색 수행
        chroma_retriever = chroma_store.as_retriever(search_kwargs={"k": k})
        chroma_docs = chroma_retriever.get_relevant_documents(query)

        # Chroma 결과 평가
        if self.evaluate_results(chroma_docs, query):
            return chroma_docs

        # Chroma 결과가 만족스럽지 않은 경우 FAISS 검색 수행
        self.logger.info("Chroma results unsatisfactory. Falling back to FAISS.")
        faiss_docs = faiss_store.similarity_search(query, k=k)

        return faiss_docs

    def evaluate_results(self, docs, query):
        # 결과 평가 로직 (예시)
        if not docs:
            return False
        relevance_score = sum(query.lower() in doc.page_content.lower() for doc in docs)
        return relevance_score >= len(query.split()) // 2  # 쿼리 단어의 절반 이상이 포함되면 만족

    def save_local(self, path):
        self.logger.warning("Saving FAISS indexes locally.")
        if self.general_faiss:
            self.general_faiss.save_local(f"{path}_general")
        if self.law_faiss:
            self.law_faiss.save_local(f"{path}_law")
        self.logger.warning("Note: Chroma data is not being saved in this operation.")

    def load_local(self, path):
        self.logger.warning("Loading FAISS indexes from local storage.")
        self.general_faiss = FAISS.load_local(f"{path}_general", self.embedding_model)
        self.law_faiss = FAISS.load_local(f"{path}_law", self.embedding_model)
        self.logger.warning("Note: Chroma data is not being loaded in this operation.")

    def clean_existing_documents(self):
        self.document_hashes.clear()
        for doc_list in [self.general_documents, self.law_documents]:
            unique_docs = []
            for doc in doc_list:
                doc_hash = self.hash_document(doc)
                if doc_hash not in self.document_hashes:
                    doc.page_content = self.clean_text(doc.page_content)
                    unique_docs.append(doc)
                    self.document_hashes.add(doc_hash)
            doc_list[:] = unique_docs

        self.create_vector_store(is_law_related=False)
        self.create_vector_store(is_law_related=True)
        self.logger.info("Existing documents cleaned and vector stores recreated.")