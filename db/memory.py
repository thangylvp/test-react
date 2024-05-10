import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import yaml
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import DistanceStrategy

class retrieval():
    def __init__(self, cfg) -> None:
        print(cfg['db'])
        assert cfg['db']['dist_type'] in ["l2", "cosine"]
        db_link = cfg.get('db',{}).get('db_link', None)
        self.db_link = db_link
        self.dist_type = cfg['db']['dist_type']
        self._build_embedding(cfg['db']['emb_model'])
        self._build_db()
        
    def _build_embedding(self, emb_model):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=emb_model, 
            model_kwargs={"trust_remote_code":True}, 
            encode_kwargs={"batch_size": 64},
        )
    
    def _build_db(self):
        if os.path.exists(self.db_link):
            print("Load db from disk", self.db_link)
            self.db = FAISS.load_local(self.db_link, self.embeddings, allow_dangerous_deserialization=True)
        else:
            trigger_texts = ["what is the capital of China?", "how to implement quick sort in python?", "Beijing", "sorting algorithms", "China", "Hanoi is capital of Vietnam"]
            if self.dist_type == "cosine":
                self.db = FAISS.from_texts(
                    trigger_texts, 
                    self.embeddings,
                    distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT,
                    normalize_L2=True,
                )
            elif self.dist_type == "l2":
                self.db = FAISS.from_texts(
                    trigger_texts, 
                    self.embeddings
                )
            self._save()

    def _save(self):
        self.db.save_local(self.db_link)

    def query(self, text, bm25_rerank=False, top_k=5):
        docs = self.db.similarity_search_with_score(text,k=50)
        all_docs = [item[0].page_content for item in docs]
            
        if bm25_rerank:
            retriever = BM25Retriever.from_texts(all_docs)
            result = retriever.get_relevant_documents(text)
            all_docs = [item.page_content for item in result]
    
        return all_docs[:top_k]

    def add(self, list_facts):
        self.db.add_texts(list_facts)

        

if __name__ == "__main__":
    with open("configs/naive_rag.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    test = retrieval(cfg=cfg)

    print("start query")
    tmp = test.query("what is the capital of China?", bm25_rerank=False) 
    print(tmp)
    # for item in tmp:
    #     print(item)
    # loader = TextLoader("../preprocess/CFA_ProgramCurriculumLevelI_Volume1_QuantitativeMethodsEconomics_sec1_1.md")
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # db = FAISS.from_documents(docs, embeddings)
    # print(db.index.ntotal)


    # query = "Holding Period Return"
    # docs = db.similarity_search(query)

    # for item in docs:
    #     print(item)
    #     print("----------------")


    # print()
    # print()
    # db.save_local("faiss_index")
    # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # docs = new_db.similarity_search(query)

    # for item in docs:
    #     print(item)
    #     print("----------------")

