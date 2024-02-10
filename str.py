import streamlit as st
import time
from inverted_index import InvertedIndex  
from catalog_lm_implementation import CatalogLM
from synonyms import SynonymGraph
import json
import networkx as nx
import matplotlib.pyplot as plt


# Initialize index and model
index = InvertedIndex()
model = CatalogLM()
model.load('catalog_lm.pt')
synonyms = SynonymGraph()

st.title("Catalog Indexer Demo")

# Indexing page
st.header("Catalog Indexing")

catalog_file = st.file_uploader("Upload Catalog JSON", type=['json']) 

def load_json(file):
  return json.load(file)

def visualize(graph):
  G = nx.DiGraph()
  G.add_edges_from(graph)
  pos = nx.spring_layout(G)

  nx.draw(G, pos, with_labels=True)

  return plt

if st.button("Index Catalog"):
    if catalog_file is not None:
        catalog = load_json(catalog_file)
        start = time.time()
        for record in catalog:
            index.add_doc(synonyms.normalize(record))
            model.index_record(synonyms.normalize(record))
        end = time.time()
        
        st.success(f"Indexed {len(catalog)} records in {end-start:.2f}s")
        st.metric("Indexing Throughput", len(catalog)/(end-start))

# Search Page
st.header("Search Catalog")  

# Free text search
query = st.text_input("Enter search query")
if st.button("Search"):
    results = index.search(query)
    st.write(results)
    
# Structured search 
filters = st.multiselect("Filter by attributes", options=["color", "category", "price"])
fetch = st.button("Fetch Products")
if fetch:
    results = model.search(filters)
    st.write(results)

# Synonyms Page    
st.header("Manage Synonyms")
term = st.text_input("Enter term")    
synonym = st.text_input("Enter synonym")

if st.button("Add Synonym"):
    synonyms.add(term, synonym)
    st.success(f"Added {synonym} as synonym for {term}")

if st.checkbox("Show Graph"):
    fig = visualize(synonyms.graph) 
    st.graphviz_chart(fig)