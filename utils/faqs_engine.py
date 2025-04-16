import openai
import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

class FAQEngine:
    def __init__(self, faqs_df, openai_api_key):
        openai.api_key = openai_api_key  # Set the API key globally
        self.faqs_df = faqs_df
        self.faq_questions = faqs_df['User Query'].tolist()
        self.faq_answers = faqs_df['Product Responses'].tolist()
        self.index = None
        self.embeddings = None
        self._build_index()

    def _embed(self, texts):
        # Use the new API method for creating embeddings
        response = openai.embeddings.create(  # Updated usage for openai>=1.0.0
            input=texts,
            model="text-embedding-ada-002"
        )
        # Access the 'embedding' attribute for each item in response.data
        embeddings = [embedding.embedding for embedding in response.data]  # Corrected access
        return embeddings

    def _build_index(self):
        # Convert list of embeddings to NumPy array
        self.embeddings = np.array(self._embed(self.faq_questions))
        # Normalize the embeddings (if required for your use case)
        normalized = normalize(self.embeddings)
        # Create a FAISS index
        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)

    def search(self, query, k=1):
        query_vec = self._embed([query])
        query_vec_norm = normalize(np.array(query_vec))  # Convert query embedding to NumPy array
        scores, indices = self.index.search(query_vec_norm, k)
        return [(self.faq_questions[i], self.faq_answers[i]) for i in indices[0]]
