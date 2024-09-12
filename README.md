# Spelling Correction Using LLM and Knowledge Retrieval

This script utilizes a combination of large language models (LLM), rule-based spell checkers, and knowledge-based retrieval to correct spelling mistakes in sentences. The correction process involves both a grammar tool (`language_tool_python`) and a retrieval-augmented generation (RAG) approach using a pre-trained model from the Hugging Face `transformers` library.

## Features

- Corrects common spelling mistakes using a pre-trained language model (`GPT-Neo`).
- Utilizes a knowledge base to retrieve similar spelling corrections and provide context for the model.
- Includes an initial pass for spelling correction using the `language_tool_python` library.
- Output is filtered to ensure the correction maintains the original meaning of the text.

## Requirements

Before you can run this project, ensure that the following Python packages are installed:

- `language_tool_python`
- `transformers`
- `scikit-learn`
- `torch`

