# CQ-RAG

This repository contains the core implementation of "CQ-RAG: Compiling Competency Questions into Evidence Contracts for Multi-Hop Question Answering".
The entry point is [cq_rag/__main__.py](cq_rag/__main__.py), with [pipeline.py](cq_rag/pipeline.py) coordinating CQ-DAG planning, knowledge extraction, retrieval, evidence reranking, reformulation, question answering and evaluation.
With Python 3.10 or newer, install dependencies using `python -m pip install -r requirements.txt`