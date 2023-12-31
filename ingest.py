"""
Ingest rem sqlite3 database to RAGchain
We use local json linker, pickle DB, and Chroma Vector DB.
Plus, I'll use openai embedding for ingest, but you can use any embedding you want.
"""
import os
from datetime import datetime, timedelta

import chromadb
import click
from RAGchain.DB import PickleDB
from RAGchain.pipeline import BasicIngestPipeline
from RAGchain.preprocess.loader.rem_loader import RemLoader
from RAGchain.retrieval import VectorDBRetrieval
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.vectorstore import ChromaSlim
from dotenv import load_dotenv

PICKLE_DB_PATH = 'DB/pickle.pkl'
CHROMA_DB_PATH = 'Chroma/'

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

    db = PickleDB(PICKLE_DB_PATH)
    vectordb = ChromaSlim(
        client=chromadb.PersistentClient(path=CHROMA_DB_PATH),
        collection_name='rem',
        embedding_function=EmbeddingFactory('openai', device_type='mps').get()
    )

    loader = RemLoader(db_path, [datetime.now() - timedelta(minutes=ingest_minutes), datetime.now()])
    retrieval = VectorDBRetrieval(vectordb)

    pipeline = BasicIngestPipeline(file_loader=loader, db=db, retrieval=retrieval)
    pipeline.run()


if __name__ == '__main__':
    main()
