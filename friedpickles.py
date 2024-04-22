import pickle
from langchain.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from fuzzywuzzy import fuzz
import re


os.environ["PINECONE_API_KEY"] = "2b4c3275-09e8-4c20-bc92-4441a136e3d5"

with open("pages", "rb") as fp: 
    pages = pickle.load(fp)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

newvectorstore = Pinecone.from_existing_index("friedpicklesquotes", hf)

def get_page(quote, pages):
    num_pages = len(pages)
    max_score = 0
    max_page = -1

    quote_words = re.findall(r'\b\w+\b', quote.lower())

    for i in range(num_pages):
        page_text = pages[i].lower()
        score = 0

        for word in quote_words:
            if word in page_text:
                score += 1

        normalized_score = score / len(quote_words)

        if normalized_score > max_score:
            max_score = normalized_score
            max_page = i + 2

        if i < num_pages - 1:
            combined_pages = (pages[i] + pages[i + 1]).lower()
            combined_score = 0

            for word in quote_words:
                if word in combined_pages:
                    combined_score += 1

            normalized_combined_score = combined_score / len(quote_words)

            if normalized_combined_score > max_score:
                max_score = normalized_combined_score
                max_page = i + 2 

    return max_page

def get_quote(prompt):
    quotes = []
    retrieved_docs = newvectorstore.similarity_search(prompt)
    for doc in retrieved_docs:
        page_num = get_page(doc.page_content, pages)
        quotes.append({"Quote": doc.page_content, "Page":page_num})
    return quotes

