import bs4
from datasets import hub

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rnd.LLMUTILS as LLMU


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_Vector_DB_Results(pdf_file,query):
    MODEL_NAME = "gemma:2b"
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    print(docs)
    model = ChatOllama(temperature=0.0, model=MODEL_NAME)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_documents = text_splitter.split_documents(docs)
    print(splitted_documents)
    vectorstore = LLMU.create_embeddings_and_store(splitted_documents, OllamaEmbeddings(model=MODEL_NAME))
    print("1", vectorstore)
    # 4-create retriver
   # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")


    print("BEFORE FORMING ChaIN")
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    print(rag_chain)


#get_Vector_DB_Results("C:\\dataset\\java.pdf" ,"what are agents in llm")





