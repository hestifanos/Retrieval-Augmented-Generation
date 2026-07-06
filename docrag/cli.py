"""Interactive retrieval from the terminal."""
import argparse

from docrag.pipeline import RagPipeline


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your documents")
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    pipeline = RagPipeline(args.docs)
    print(f"Indexed documents from {args.docs}. Type a question, or 'quit'.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        for chunk, score in pipeline.retrieve(question, k=args.k):
            print(f"\n[{chunk.doc_id}#{chunk.chunk_id}] score={score:.3f}\n{chunk.text[:300]}")


if __name__ == "__main__":
    main()
