# vLLM Hack — Multimodal RAG

A multimodal Retrieval-Augmented Generation system built with vision-language models, hybrid search, reranking, and RAGAs-based evaluation.

## Track

**Build multimodal RAG with vision-language models.** Evaluate retrieval quality with RAGAs metrics. Implement reranking and hybrid search.

## Features

- **Multimodal ingestion** — index documents containing both text and images (PDFs, slides, screenshots, diagrams).
- **Vision-language model (VLM) backbone** — answers grounded in retrieved text *and* image context.
- **Hybrid search** — combines dense vector retrieval with sparse lexical (BM25) for higher recall on rare terms and exact matches.
- **Reranking** — cross-encoder reranker over the fused candidate set to sharpen top-k precision before the VLM sees it.
- **RAGAs evaluation** — faithfulness, answer relevancy, context precision, and context recall reported on a held-out eval set.

## Architecture

```
                ┌──────────────┐
   query ─────► │  Hybrid      │  dense (embeddings) + sparse (BM25)
                │  Retriever   │
                └──────┬───────┘
                       │  top-k candidates (text + image refs)
                       ▼
                ┌──────────────┐
                │  Reranker    │  cross-encoder, returns top-n
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │  VLM         │  text + images → grounded answer
                └──────┬───────┘
                       ▼
                    answer
                       │
                       ▼
                ┌──────────────┐
                │  RAGAs       │  faithfulness, relevancy, precision, recall
                └──────────────┘
```

## Stack

- **Serving** — vLLM for the VLM backend
- **Embeddings** — open-source multimodal embedding model
- **Vector store** — for dense retrieval
- **BM25** — for sparse retrieval
- **Reranker** — cross-encoder
- **Eval** — [RAGAs](https://github.com/explodinggradients/ragas)

## Quick start

```bash
# install
pip install -r requirements.txt

# index a corpus of mixed text + images
python -m src.ingest --input ./data/corpus

# ask a question
python -m src.ask "What does the diagram on page 4 show?"

# run evaluation
python -m src.eval --dataset ./data/eval.jsonl
```

## Repository layout

```
src/
  ingest.py        # parsing, chunking, embedding, indexing
  retrieve.py      # hybrid retrieval (dense + BM25)
  rerank.py        # cross-encoder reranking
  generate.py      # VLM call with text + image context
  eval.py          # RAGAs evaluation harness
data/
  corpus/          # source documents (text + images)
  eval.jsonl       # eval set (question, ground truth, contexts)
```

## Evaluation

We report the four standard RAGAs metrics:

| Metric              | What it measures                                       |
| ------------------- | ------------------------------------------------------ |
| Faithfulness        | Is the answer grounded in retrieved context?           |
| Answer relevancy    | Does the answer address the question?                  |
| Context precision   | Are the retrieved chunks actually relevant?            |
| Context recall      | Did we retrieve everything needed for the answer?      |

Ablations compare: dense-only vs. hybrid, with vs. without reranker, text-only vs. multimodal context.

## License

MIT — see [LICENSE](./LICENSE).
