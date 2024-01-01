import chromadb
import click
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import VectorDBRetrieval
from RAGchain.schema import Passage, RAGchainChatPromptTemplate
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.vectorstore import ChromaSlim
from langchain_community.chat_models import ChatOpenAI

CHROMA_DB_PATH = 'Chroma/'
PROMPT = RAGchainChatPromptTemplate.from_messages([
    ("user", "Please answer {question}"),
    ("system", "--------------- Below is the text that\'s been on screen recently. --------------"),
    ("system", "{passages}"),
    ("system", "Please answer the query using the provided information about what has been on the scren recently: "
               "{question}\nDo not say anything else or give any other information. Only answer the question : "
               "{question}"),
])


@click.command()
@click.option('--query', help='Query you want to ask to LLM.')
def main(query):
    print(f"Question : {query}")

    vectordb = ChromaSlim(
        client=chromadb.PersistentClient(path=CHROMA_DB_PATH),
        collection_name='rem',
        embedding_function=EmbeddingFactory('openai', device_type='mps').get()
    )
    pipeline = BasicRunPipeline(VectorDBRetrieval(vectordb), ChatOpenAI(), prompt=PROMPT)
    answers, passages, scores = pipeline.get_passages_and_run([query])
    print(f"Answer : {answers[0]}")
    print(f"Passages : {Passage.make_prompts(passages[0])}")


if __name__ == '__main__':
    main()
