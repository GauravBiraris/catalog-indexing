import re
import nltk
import networkx as nx
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from annoy import AnnoyIndex
from pydictionary import PyDictionary

class Indexer:

  def __init__(self):
    schema = Schema(id=ID(stored=True), text=TEXT) 
    self.index = create_in("indexdir", schema)  
    self.graph = nx.Graph()
    self.annoy = AnnoyIndex(100) 
    self.dictionary = PyDictionary()
    self.wordnet = nltk.corpus.wordnet

  def index_catalog(self, catalog):
    writer = self.index.writer()
    for item in catalog.items:
      self.index_item(writer, item)
    writer.commit()

  def index_item(self, writer, item):
    tokens = self.tokenize(item.data)
    indexes = self.map_to_indexes(tokens)
    self.annoy.add_item(item.id, indexes) 
    self.graph.add_node(item.id)
    writer.add_document(id=item.id, text=str(indexes))

  def tokenize(self, text):
    return nltk.word_tokenize(text.lower())   

  def map_to_indexes(self, tokens):
    indexes = []
    for token in tokens:
      synsets = self.wordnet.synsets(token)
      lemmas = set() 
      for synset in synsets:
        lemmas.update(synset.lemma_names())

      if len(lemmas) > 1:  
        index = token + '_' + str(len(indexes))  
      else:
        index = token

      indexes.append(index)

    return indexes
  
  def search(self, query):
    tokens = self.tokenize(query)
    indexes = self.map_indexes(tokens)
    results = [hit['id'] for hit in self.index.searcher().search(indexes)]
    ordered = nx.pagerank(self.graph, results)
    return ordered[:10]

  def map_indexes(self, tokens):
    indexes = []
    for token in tokens:
      if token in indexes:
        continue

      synsets = self.wordnet.synsets(token)
      lemmas = set()
      for synset in synsets:
        lemmas.update(synset.lemma_names())

      if len(lemmas) > 1:
        related = [i for i in indexes if i.startswith(token)]  
        indexes.extend(related)
        index = token + '_' + str(len(indexes))
      else:
        index = token

      indexes.append(index)

    return indexes
