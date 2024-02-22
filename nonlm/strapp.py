import streamlit as st
from indexer import Indexer
from catalog import Catalog

catalog = Catalog()
indexer = Indexer()

st.title('My Catalog App')

# Index catalog
st.write('Indexing catalog...')
indexer.index_catalog(catalog)
st.write('Done!')

# Add item
st.header('Add New Item')
name = st.text_input('Name')
desc = st.text_area('Description')
if st.button('Add'):
  item = Item(name, desc) 
  indexer.index_item(item)
  catalog.add_item(item)
  st.success('Item added!')

# Update item
items = {item.id: item.name for item in catalog.items}
selected_id = st.selectbox('Select item to update', options=items.keys())  
name = st.text_input('New name', value=catalog.get_item(selected_id).name)
desc = st.text_area('New description', value=catalog.get_item(selected_id).desc)
if st.button('Update'):
  item = catalog.get_item(selected_id)
  item.name = name
  item.desc = desc
  indexer.update_item(item)
  st.success('Item updated!')

# Delete item
selected_id = st.selectbox('Select item to delete', options=items.keys())
if st.button('Delete'):
  indexer.delete_item(selected_id)
  catalog.delete_item(selected_id)
  st.warning('Item deleted!')

# Search 
query = st.text_input('Search')  
if query:
  results = indexer.search(query)
  for rank, item in enumerate(results):
    st.write(f'{rank+1}. {item.name} ({item.id})')