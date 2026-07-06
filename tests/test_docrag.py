import json
from pathlib import Path

from docrag.chunker import chunk_document
from docrag.evaluate import evaluate
from docrag.pipeline import RagPipeline
from docrag.retriever import TfidfRetriever, tokenize

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


def test_chunker_splits_and_overlaps():
    text = " ".join(f"Sentence number {i} adds some content here." for i in range(40))
    chunks = chunk_document("doc", text, max_chars=200, overlap_sentences=1)
    assert len(chunks) > 1
    # Overlap: the last sentence of chunk 0 also appears in chunk 1.
    last_sentence = chunks[0].text.rstrip(".").split(". ")[-1]
    assert last_sentence in chunks[1].text


def test_tokenizer_drops_stopwords():
    assert tokenize("The cat and the hat") == ["cat", "hat"]


def test_retriever_ranks_relevant_chunk_first():
    chunks = chunk_document("a", "Bananas are yellow fruit rich in potassium.") + \
             chunk_document("b", "Volcanoes erupt molten rock called lava.")
    top = TfidfRetriever(chunks).search("fruit potassium", k=1)
    assert top[0][0].doc_id == "a"


def test_pipeline_recall_on_sample_corpus():
    pipeline = RagPipeline(DOCS)
    cases = json.loads((DOCS / "eval_cases.json").read_text())
    metrics = evaluate(pipeline, cases, k=3)
    assert metrics["recall_at_k"] >= 0.85
    assert metrics["mrr"] >= 0.75


def test_prompt_contains_citations_and_question():
    prompt = RagPipeline(DOCS).build_prompt("Who proposed the web?")
    assert "internet_history#" in prompt
    assert "Who proposed the web?" in prompt
