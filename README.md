# code-help

## Setup
Ensure that the local Python version is >= 3.8. Install the required libraries:

```bat
pip install -r requirements.txt
```

### How to Run
```bash
python code_help.py
```

### Reference
Code Referenced from https://github.com/Chitti-Ankith/Story-QA-using-GPT4All/tree/main

### What Happens
- Inputs a file specified by the user (or a default file) and loads it into EvaDB as a string. 
- Then, it creates code embeddings using a transformers model (all-MiniLM-L6-v2).
- Creates an index using Qdrant to help with searching.
- Reads question from user.
- Runs a similarity search on the question throught out the embeddings table and returns a list of the top 5 most similar entries.
- Uses a GPT4All model (ggml-model-gpt4all-falcon-q4_0.bin) to generate an answer to the user's question based upon those 5 entries.
