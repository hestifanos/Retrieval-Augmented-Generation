# doc-rag

A retrieval-augmented generation pipeline built from first principles. It
chunks a folder of text documents, indexes them with a TF-IDF vectorizer
implemented by hand, and produces grounded prompts with chunk-level citations
ready to send to any LLM. It ships with an evaluation harness that measures
recall@k and mean reciprocal rank against a labelled question set.

## Why build TF-IDF from scratch

Most RAG tutorials call an embeddings API and a vector database and stop
there. I wanted to demonstrate an understanding of what happens underneath:
tokenization, log-scaled term frequency, smoothed inverse document frequency,
vector normalization and cosine scoring are all written out in
docrag/retriever.py in about sixty lines. The retriever interface is small,
so swapping in dense embeddings later is a drop-in change.

## Layout

    docrag/chunker.py     sentence-aware chunking with configurable overlap
    docrag/retriever.py   hand-rolled TF-IDF and cosine similarity search
    docrag/pipeline.py    document loading, indexing, prompt assembly
    docrag/evaluate.py    recall@k and MRR over a labelled eval set
    docrag/cli.py         interactive retrieval from the terminal
    sample_docs/          small corpus plus eval_cases.json

## Usage

    python -m docrag.cli --docs sample_docs
    python -m docrag.evaluate --docs sample_docs -k 3

On the included corpus the pipeline scores 1.0 recall@3 and 1.0 MRR, and the
test suite enforces a floor on both metrics so retrieval regressions fail CI.

## Tests

    pytest -v

## Design decisions

- Chunking respects sentence boundaries and repeats the trailing sentence of
  each chunk into the next one, so answers that straddle a chunk boundary
  remain retrievable.
- Answer generation is deliberately decoupled from retrieval. build_prompt
  returns a grounded prompt with citation markers that can be piped to any
  model, keeping the retrieval layer independently testable.
- Evaluation is part of the project, not an afterthought. Adding a document
  means adding eval cases for it.

## Next steps

- Dense embedding retriever behind the same interface, with an A/B eval
- BM25 scoring as a middle ground
- PDF and Markdown ingestion
