"""Small CLI to load TruthfulQA, embed texts, and store into ChromaDB.

Usage examples:
  python sandbox.py --download                # download dataset then load
  python sandbox.py --path data/truthfulqa/validation.jsonl

Environment:
  - To use Hugging Face Inference API set `HF_API_TOKEN` env var.
  - Requires `chromadb` and either `huggingface-hub` or `sentence-transformers`.
"""

import os
import argparse
import math
import re
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import requests
import chromadb
from chromadb.utils import embedding_functions

from data_fetchers import load_dataset_into_memory

# Configuration
ENDPOINT_URL = "https://yyirs15si60zuagp.us-east-1.aws.endpoints.huggingface.cloud" # Endpoint URL + version
HF_TOKEN = os.getenv("HF_API_TOKEN") # Your Hugging Face Hub token from hf.co/settings/tokens

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

api_url = f"https://router.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()


from huggingface_hub import InferenceClient

client = InferenceClient(model= ENDPOINT_URL, token=HF_TOKEN)
 
from huggingface_hub import HfApi


result = client.feature_extraction(
    "Explain transformers simply"
)

print(result)

def get_embeddings(texts: List[str]) -> List[List[float]]:
	results = client.feature_extraction(texts)
	return results


dataset = load_dataset_into_memory("truthfulqa")

# 
print(len(dataset))
print(dataset[0].keys())
df = pd.DataFrame.from_dict({i: dataset[i] for i in range(len(dataset))}).T
print(df.head())

text_cols = ['question','best_answer','correct_answers','incorrect_answers']
df_correct = df[['question','correct_answers']]
df_correct = df_correct.explode(column='correct_answers')
df_incorrect = df[['question','incorrect_answers']]
df_incorrect = df_incorrect.explode(column='incorrect_answers')
print(df_correct.head())
print(df_incorrect.head())



# Save the processed dataset locally
dataset_with_embeddings.save_to_disk("./embedded_dataset")

# Or push directly to Hugging Face Hub
dataset_with_embeddings.push_to_hub("your-username/squad-embeddings")

query(texts)

try:
	items = dataset_loader.load_dataset_into_memory("truthfulqa", path=os.path.join("data", "truthfulqa", "validation.jsonl"))
	print(f"Loaded {len(items)} items from TruthfulQA validation split")
	if items:
		print("Example item:", items[0])
		print("Items keys:\n", "\n".join(items[0].keys()))
except Exception:
	items = []


def extract_sentences_from_item(item: Dict) -> List[str]:
	# Collect textual fields from dataset item and split into sentences.
	texts: List[str] = []
	for k, v in item.items():
		if isinstance(v, str) and v.strip():
			texts.append(v.strip())
		# handle lists of strings (e.g., answers)
		if isinstance(v, list):
			for el in v:
				if isinstance(el, str) and el.strip():
					texts.append(el.strip())

	# join and split into sentences conservatively
	joined = "\n".join(texts)
	# split on sentence boundaries
	parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', joined) if s.strip()]
	return parts if parts else ([joined] if joined else [])

extract = extract_sentences_from_item(items[0]) if items else []


def get_embeddings_hf_or_local(texts: List[str], hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
							   batch_size: int = 64):
	"""Try Hugging Face Inference API first; fall back to SentenceTransformer local model."""
	# Lazy imports to keep dependency foot-print small on import
	hf_token = os.environ.get("HF_API_TOKEN")
	if hf_token:
		# Try modern client, then HTTP fallback
		try:
			# Newer versions expose InferenceClient
			from huggingface_hub import InferenceClient

			client = InferenceClient(token=hf_token)
			embs = []
			for i in range(0, len(texts), batch_size):
				batch = texts[i : i + batch_size]
				# embeddings pipeline: use the model via `embeddings` pipeline
				out = client.embeddings(model=hf_model, inputs=batch)
				for item in out:
					v = item.get("embedding") if isinstance(item, dict) and "embedding" in item else item
					embs.append([float(x) for x in v])
			return embs
		except Exception:
			try:
				from huggingface_hub import InferenceApi

				api = InferenceApi(repo_id=hf_model, token=hf_token, task="feature-extraction")
				embs = []
				for i in range(0, len(texts), batch_size):
					batch = texts[i : i + batch_size]
					out = api(batch)
					for vec in out:
						if not vec:
							embs.append([0.0])
							continue
						if isinstance(vec[0], list):
							avg = [float(x) for x in [sum(col) / len(col) for col in zip(*vec)]]
							embs.append(avg)
						else:
							embs.append([float(x) for x in vec])
				return embs
			except Exception:
				# final fallback: direct HTTP call to HF Inference API
				try:
					import requests

					headers = {"Authorization": f"Bearer {hf_token}"}
					embs = []
					for i in range(0, len(texts), batch_size):
						batch = texts[i : i + batch_size]
						resp = requests.post(f"https://api-inference.huggingface.co/models/{hf_model}", headers=headers, json={"inputs": batch})
						resp.raise_for_status()
						out = resp.json()
						for vec in out:
							if not vec:
								embs.append([0.0])
								continue
							if isinstance(vec[0], list):
								avg = [float(x) for x in [sum(col) / len(col) for col in zip(*vec)]]
								embs.append(avg)
							else:
								embs.append([float(x) for x in vec])
					return embs
				except Exception:
					print("HF inference API unavailable or failed; falling back to local SentenceTransformer.")

	# fallback: local SentenceTransformer
	try:
		from sentence_transformers import SentenceTransformer
		import numpy as np

		model = SentenceTransformer(hf_model)
		arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
		embs = arr.tolist()
		return embs
	except Exception as e:
		raise RuntimeError("No embedding method available: set HF_API_TOKEN or install sentence-transformers") from e


def upsert_to_chroma(collection_name: str, ids: List[str], embeddings: List[List[float]], docs: List[str], metadatas: List[Dict], persist_directory: Optional[str] = None):
	# create client; if persist_directory provided use it
	client = chromadb.Client()
	try:
		coll = client.get_collection(name=collection_name)
	except Exception:
		coll = client.create_collection(name=collection_name)

	# Insert in a single call (Chroma handles batch sizes internally)
	coll.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
	return coll


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--download", action="store_true", help="Download TruthfulQA into data/truthfulqa/")
	parser.add_argument("--path", type=str, default=None, help="Path to a .jsonl file to load")
	parser.add_argument("--hf_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face model id for embeddings")
	parser.add_argument("--batch", type=int, default=256, help="Batch size for embedding requests")
	parser.add_argument("--collection", type=str, default="truthfulqa", help="Chroma collection name")
	args = parser.parse_args()

	if args.download:
		print("Downloading TruthfulQA (may take a while)...")
		dataset_loader.download_dataset("truthfulqa")

	items = dataset_loader.load_dataset_into_memory("truthfulqa") if args.path is None else dataset_loader.load_dataset_into_memory("truthfulqa", path=args.path)
	print(f"Loaded {len(items)} items from TruthfulQA")

	# extract sentences from all items
	texts: List[str] = []
	src_idx: List[int] = []
	for idx, it in enumerate(items):
		sents = extract_sentences_from_item(it)
		for s in sents:
			texts.append(s)
			src_idx.append(idx)

	print(f"Extracted {len(texts)} text chunks/sentences to embed")

	hf_token = os.environ.get("HF_API_TOKEN")
	# embed in batches and insert into chroma in batches
	batch = args.batch
	chroma_ids: List[str] = []
	chroma_embs: List[List[float]] = []
	chroma_docs: List[str] = []
	chroma_meta: List[Dict] = []

	total = len(texts)
	for i in range(0, total, batch):
		chunk_texts = texts[i : i + batch]
		embs = get_embeddings_hf_or_local(chunk_texts, args.hf_model, hf_token, batch_size=64)

		# prepare ids/metas/docs
		for j, t in enumerate(chunk_texts):
			idx_global = i + j
			cid = f"ttqa-{idx_global}"
			chroma_ids.append(cid)
			chroma_embs.append(embs[j])
			chroma_docs.append(t)
			chroma_meta.append({"source_item": src_idx[idx_global]})

		print(f"Prepared batch {i}..{i+len(chunk_texts)} (embeddings retrieved)")

	# Insert into Chroma in one call; if dataset is huge you can change to smaller batches
	coll = upsert_to_chroma(args.collection, chroma_ids, chroma_embs, chroma_docs, chroma_meta)
	print(f"Inserted {len(chroma_ids)} vectors into Chroma collection '{args.collection}'")


if __name__ == '__main__':
	main()
