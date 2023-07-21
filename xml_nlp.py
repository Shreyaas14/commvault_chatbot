import xml.etree.ElementTree as ET

import numpy as np
import torch 
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

#XML PARSING
tree = ET.parse('commvault_essential_docs.xml')
root = tree.getroot()

urls = []

ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

for url in root.findall('sitemap:url', ns):
    loc = url.find('sitemap:loc', ns)
    if loc is not None and loc.text is not None:
        urls.append(loc.text)


#pretrained modele tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_embedding(message):
    input_ids = torch.tensor([tokenizer.encode(message, add_special_tokens=True)])

    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]

    sentence_embedding = torch.mean(last_hidden_states, dim = 1)

    return sentence_embedding.numpy()

#article_embeddings = []
#for i, url in enumerate(urls):
    #article_embeddings.append(get_embedding(url))
    #if i % 100 == 0:
        #print(f"Computed embeddings for {i} articles.")

#article_embeddings_np = np.array(article_embeddings)
#np.save('article_embeddings.npy', article_embeddings_np)

def get_most_similar_urls(user_query, top_n=5):
    article_embeddings_np_final = np.load('article_embeddings.npy')
    user_embedding = get_embedding(user_query)
    
    similarities = np.array([cosine_similarity(user_embedding.reshape(1, -1), url_embedding.reshape(1, -1))[0][0] for url_embedding in article_embeddings_np_final])
    
    most_similar_url_indices = similarities.argsort()[-top_n:][::-1]
    
    # Convert numpy array to a list
    most_similar_url_indices = most_similar_url_indices.tolist()
    
    return [urls[i] for i in most_similar_url_indices]
