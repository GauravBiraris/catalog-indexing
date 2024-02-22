from dataclasses import dataclass

@dataclass 
class Item:
  id: str
  data: dict

  
class Catalog:

  def __init__(self):
    self.items = []

  def add_item(self, item):
    self.items.append(item)

  def get_item(self, item_id):
    for item in self.items:
      if item.id == item_id:
        return item
    return None  

  def remove_item(self, item_id):
    item = self.get_item(item_id)
    if item:
      self.items.remove(item)

  def search(self, query):
    results = []
    for item in self.items:
      if query in item.data.values():
        results.append(item)
    return results
