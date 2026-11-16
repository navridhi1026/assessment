from backend.embeddings import SentenceTransformerEmbedder
from backend.llm import OllamaClient
from backend.vector_store import ChromaVectorStore


NO_ANSWER = "I couldn't find relevant information in the uploaded documents."


class QuestionAnsweringService:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: SentenceTransformerEmbedder,
        llm: OllamaClient,
        relevance_distance_threshold: float,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.relevance_distance_threshold = relevance_distance_threshold

    def ask(self, question: str) -> dict:
        if not self.vector_store.has_documents():
            return {"answer": NO_ANSWER, "sources": []}
        matches = self.vector_store.search(self.embedder.embed_query(question))
        matches = [
            item
            for item in matches
            if item["distance"] <= self.relevance_distance_threshold
        ]
        if not matches:
            return {"answer": NO_ANSWER, "sources": []}

        answer = self.llm.answer(question, matches)
        sources = [
            {
                "filename": item["filename"],
                "page_number": item["page_number"],
                "excerpt": item["text"][:500],
                "distance": item["distance"],
            }
            for item in matches
        ]
        return {"answer": answer, "sources": sources}
