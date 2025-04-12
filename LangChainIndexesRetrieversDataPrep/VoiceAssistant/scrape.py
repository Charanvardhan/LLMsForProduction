import os 
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

import re


from dotenv import load_dotenv
load_dotenv()


my_activeloop_org_id = "charanvardhan"
my_activeloop_dataset_name = "VoiceAssistant-embeddings"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def get_documents_url(url):
    return [
    '/docs/huggingface_hub/guides/overview',
    '/docs/huggingface_hub/guides/download',
    '/docs/huggingface_hub/guides/upload',
    '/docs/huggingface_hub/guides/hf_file_system',
    '/docs/huggingface_hub/guides/repository',
    '/docs/huggingface_hub/guides/search',
    ]

def construct_url(base_url, relative_paath):
    return base_url + relative_paath


def scrape_page_content(url):

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract the desired content from the page 
    text = soup.body.text.strip()

    # remove none ascii characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def scrape_all_content(base_url, relative_paths, 
                       filename):
    all_text = []
    for path in relative_paths:
        url = construct_url(base_url, path)
        text = scrape_page_content(url)
        all_text.append(text)
    

    with open(filename, 'w', encoding='utf-8') as f:
        for item in all_text:
            f.write(item + '\n')
        
    return all_text



# load and splitting the text
# def load_docs(root_dir, filename):
#     docs = []
#     try:
#         loader = TextLoader(os.path.join(root_dir, filename, encoding='utf-8'))
#         docs.extend(loader.load_and_split())
    
#     except Exception as e:
#         pass
    
#     return docs

def load_docs(root_dir, filename):
    docs = []
    try:
        file_path = os.path.join(root_dir, filename)
        loader = TextLoader(file_path, encoding='utf-8')  # ✅ fixed encoding
        # docs = loader.load()  # ✅ use .load()
        docs.extend(loader.load_and_split())
    except Exception as e:
        print(f"Error loading documents: {e}")  # ✅ helpful error message
    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def main():
    base_url = 'https://huggingface.co'

    filename = 'content.txt'
    root_dir = './'
    relative_paths = get_documents_url(base_url)
    content = scrape_all_content(base_url, relative_paths, filename)
    docs = load_docs(root_dir, filename)

    texts = split_docs(docs)

    # db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    # db.add_documents(texts)
    os.remove(filename)
    

if __name__ == "__main__":
    main()

