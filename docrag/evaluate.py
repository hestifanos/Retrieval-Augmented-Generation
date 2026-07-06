"""Retrieval evaluation: recall@k and mean reciprocal rank over a labelled set."""
import argparse
import json
from pathlib import Path

from docrag.pipeline import RagPipeline


def evaluate(pipeline: RagPipeline, cases: list, k: int = 3) -> dict:
    hits = 0
    reciprocal_ranks = []
    for case in cases:
        results = pipeline.retrieve(case["question"], k=k)
        rank = next((i + 1 for i, (c, _) in enumerate(results) if c.doc_id == case["doc_id"]), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    n = len(cases)
    return {"recall_at_k": hits / n, "mrr": sum(reciprocal_ranks) / n, "k": k, "cases": n}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("--cases", default="sample_docs/eval_cases.json")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    metrics = evaluate(RagPipeline(args.docs), json.loads(Path(args.cases).read_text()), k=args.k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
