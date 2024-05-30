# Get Started
1. Move alfworld directory and install requirements
```bash
cd ./alfworld
pip install -r requirements.txt
```

2. According to the [instruction](https://github.com/alfworld/alfworld?tab=readme-ov-file#quickstart), download the data and set environment variables 

3. Prepare OpenAI API key and put the key in ```OpenAI_api_key.txt```

4. Run RAP on ALFWorld
```bash
python main.py
```

Following are the hyper-parametes for RAP.

* num_trials: Number of recursive trials. Default is 3.
* num_steps: Maximum steps in one task. Default is 50.
* model: Model to be used in evaluation. Default is "gpt-3.5-turbo-instruct".
* output: Folder path to output logs and memory.
* emb_model: Embedding model to be used in evaluation. Default is "sentence-transformers/all-MiniLM-L6-v2".

Also, Python 3.11 is recommended for evaluation.