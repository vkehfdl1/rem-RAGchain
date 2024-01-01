"""
Ingest rem sqlite3 database to RAGchain
We use local json linker, pickle DB, and Chroma Vector DB.
Plus, I'll use openai embedding for ingest, but you can use any embedding you want.
"""
import itertools
import os
from datetime import datetime, timedelta

import chromadb
import click
from RAGchain.DB import PickleDB
from RAGchain.benchmark.answer.metrics import KF1
from RAGchain.preprocess.loader.rem_loader import RemLoader
from RAGchain.preprocess.text_splitter import TokenSplitter
from RAGchain.retrieval import VectorDBRetrieval, HybridRetrieval, BM25Retrieval
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.vectorstore import ChromaSlim
from dotenv import load_dotenv

PICKLE_DB_PATH = 'DB/pickle.pkl'
CHROMA_DB_PATH = 'Chroma/'
BM25_PATH = 'DB/bm25.pkl'

load_dotenv()


@click.command()
@click.option("--db_path",
              help="Path to rem sqlite3 database. You can see filepath in rem menu 'show my data', and find db.sqlite3 file.")
@click.option("--ingest_minutes", type=int, help='Minutes you want to ingest. Default is 5 minutes.', default=5)
def main(db_path, ingest_minutes):
    if not os.path.exists(os.path.dirname(PICKLE_DB_PATH)):
        os.makedirs(os.path.dirname(PICKLE_DB_PATH))
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
    if not os.path.exists(os.path.dirname(BM25_PATH)):
        os.makedirs(os.path.dirname(BM25_PATH))

    db = PickleDB(PICKLE_DB_PATH)
    vectordb = ChromaSlim(
        client=chromadb.PersistentClient(path=CHROMA_DB_PATH),
        collection_name='rem',
        embedding_function=EmbeddingFactory('openai', device_type='mps').get()
    )

    loader = RemLoader(db_path, [datetime.now() - timedelta(minutes=ingest_minutes), datetime.now()])
    retrieval = HybridRetrieval([VectorDBRetrieval(vectordb), BM25Retrieval(BM25_PATH)], [0.7, 0.3],
                                method='cc')

    documents = loader.load()
    ingest_documents = []
    # keep a document that has no duplicate
    for i, document in enumerate(documents):
        if i == 0:
            ingest_documents.append(document)
            continue
        # calculate similarity with a previous document using token f1 score
        kf1 = KF1()
        score = kf1._token_f1_score(document.page_content, documents[i - 1].page_content)
        if score < 0.9:
            ingest_documents.append(document)

    print(f"Total {len(documents)} documents, {len(ingest_documents)} documents will be ingested.")

    splitter = TokenSplitter(chunk_size=1024, chunk_overlap=128)
    passages = splitter.split_documents(ingest_documents)

    print(f"Total {len(passages)} passages will be ingested.")

    # ingest to db and retrieval
    db.create_or_load()
    db.save(list(itertools.chain.from_iterable(passages)))

    retrieval.ingest(list(itertools.chain.from_iterable(passages)))


if __name__ == '__main__':
    main()
