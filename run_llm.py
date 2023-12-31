import chromadb
import click
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import VectorDBRetrieval
from RAGchain.schema import Passage
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.vectorstore import ChromaSlim
from langchain_community.chat_models import ChatOpenAI

CHROMA_DB_PATH = 'Chroma/'


@click.command()
@click.option('--query', help='Query you want to ask to LLM.')
def main(query):
    print(f"Question : {query}")

    vectordb = ChromaSlim(
        client=chromadb.PersistentClient(path=CHROMA_DB_PATH),
        collection_name='rem',
        embedding_function=EmbeddingFactory('openai', device_type='mps').get()
    )
    pipeline = BasicRunPipeline(VectorDBRetrieval(vectordb), ChatOpenAI())
    answers, passages, scores = pipeline.get_passages_and_run([query])
    print(f"Answer : {answers[0]}")
    print(f"Passages : {Passage.make_prompts(passages[0])}")


if __name__ == '__main__':
    main()
