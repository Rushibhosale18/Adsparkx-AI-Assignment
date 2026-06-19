#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Ingest the knowledge base to build the ChromaDB vector store
python ingest.py
