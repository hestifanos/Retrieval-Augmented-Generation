"""End-to-end pipeline: load documents, chunk, index, answer questions."""
from pathlib import Path

from docrag.chunker import chunk_document
from docrag.retriever import TfidfRetriever


class RagPipeline:
    def __init__(self, docs_dir):
        self.docs_dir = Path(docs_dir)
        chunks = []
        for path in sorted(self.docs_dir.glob("*.txt")):
            chunks.extend(chunk_document(path.stem, path.read_text()))
        if not chunks:
            raise ValueError(f"No .txt documents found in {self.docs_dir}")
        self.retriever = TfidfRetriever(chunks)

    def retrieve(self, question: str, k: int = 3):
        return self.retriever.search(question, k=k)

    def build_prompt(self, question: str, k: int = 3) -> str:
        """Assemble a grounded prompt for any LLM. Answer generation is
        deliberately decoupled so retrieval quality can be evaluated alone."""
        context = "\n\n".join(
            f"[{c.doc_id}#{c.chunk_id}] {c.text}" for c, _ in self.retrieve(question, k)
        )
        return (
            "Answer the question using only the context below. "
            "Cite chunk ids like [doc#0]. If the context is insufficient, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
