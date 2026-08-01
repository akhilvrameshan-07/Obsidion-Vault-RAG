---
tags: [reference, ai]
aliases: [RAG Notes]
created: 2025-01-02
---
# RAG Pipeline Notes

Retrieval augmented generation combines a search step with a generation step so an AI model can answer questions using private documents instead of only its training data.

## How Retrieval Works

Every note gets split into chunks, and each chunk is converted into a vector using a local embedding model. When a question comes in, it is embedded the same way, and the vector database returns the chunks whose vectors are closest to the question's vector.

## How Generation Works

The retrieved chunks are handed to a language model along with the original question. The model is instructed to answer only using the provided chunks, and to say so if nothing relevant was found. See [[note1]] for why this project exists and [[note3]] for where the vector database actually lives.