from datasets import load_dataset
import torch
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import os

HFTOKEN = os.environ.get("HF_TOKEN", "")

class SimpleRAGNews():

	def __init__(self):

		# Note: the original dataset "permutans/fineweb-bbc-news" (subset: "CC-MAIN-2013-20")
		# has been taken down from HuggingFace. Using "SetFit/bbc-news" as a replacement.
		self.dataset = load_dataset(
			"SetFit/bbc-news",
			split="train",
			streaming=True
		)

		# load the model "ibm-granite/granite-embedding-30m-english"
		# and corresponding tokenizer using AutoModel and AutoTokenizer
		self.tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-30m-english")
		self.model = AutoModel.from_pretrained("ibm-granite/granite-embedding-30m-english")
		self.model.eval()

		self.setup_db()

		# finally, create a client to use huggingface inference
		self.client = OpenAI(
		    base_url="https://router.huggingface.co/v1",
		    api_key=HFTOKEN,
		)


	def setup_db(self):
		# take the first 100 samples from the bbc-news dataset
		# pare down the columns and only keep the "text" column
		# then convert to a list: list(ds) for easy retrieval later
		ds = self.dataset.take(100)
		ds = ds.select_columns(["text"])
		self.articles = list(ds)

		# for each entry in the dataset, call self.embed() on the text
		# save these vectors for later use
		self.embeddings = []
		for article in self.articles:
			embedding = self.embed(article["text"])
			self.embeddings.append(embedding)

		# stack into a single tensor of shape (100, hidden_dim)
		self.embeddings = torch.cat(self.embeddings, dim=0)


	def embed(self, text):
		# tokenize the text using the ibm-granite tokenizer
		tokens = self.tokenizer(
			text,
			padding=True,
			truncation=True,
			return_tensors='pt'
		)

		# pass through the embedding model
		with torch.no_grad():
			embedding = self.model(**tokens)[0][:, 0]
			embedding = torch.nn.functional.normalize(embedding, dim=1)

		return embedding


	def get_most_relevant_news_article_text(self, user_query):
		# embed the user query
		query_embedding = self.embed(user_query)  # shape: (1, hidden_dim)

		# compare against all stored embeddings using cosine_similarity
		similarities = torch.nn.functional.cosine_similarity(
			query_embedding, self.embeddings  # broadcasts over 100 rows
		)

		# find the index of the top match
		best_idx = similarities.argmax().item()

		return self.articles[best_idx]["text"]

	def summarize_article(self, article):
		# Use the HF Inference API to ask "openai/gpt-oss-20b" to summarize an article.
		completion = self.client.chat.completions.create(
			model="openai/gpt-oss-20b",
			messages=[
				{
					"role": "user",
					"content": f"Please summarize the following news article in 2-3 sentences:\n\n{article}"
				}
			]
		)
		return completion.choices[0].message.content

	def summary_for_query(self, query):
		# get the most relevant article
		article = self.get_most_relevant_news_article_text(query)
		# get a summary of the article
		summary = self.summarize_article(article)
		# return to user
		return summary



if __name__ == "__main__":

	rag = SimpleRAGNews()
	query = "california wildfires"
	news_blurb_for_user = rag.summary_for_query(query)

	print("An AI-generated summary of the most relevant article:")
	print(news_blurb_for_user)
