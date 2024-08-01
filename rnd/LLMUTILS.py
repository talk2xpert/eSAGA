#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langserve import add_routes
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory


from langchain_core.runnables import RunnablePassthrough



def load_document(pdf_file_name):
    loader = PyPDFLoader(pdf_file_name)
    return loader

def split_document(loader):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    splitted_documents = text_splitter.split_documents(loader.load())
    return  splitted_documents

def create_embeddings_and_store(splitted_documents,embedding_object):
    vectorstore = Chroma.from_documents(documents=splitted_documents, embedding=embedding_object)
    return  vectorstore

def get_document_from_vector_db_embedding(vectorstore,query,no_of_documents):
    docs=vectorstore.similarity_search(
        query, k=no_of_documents)
    return docs


def get_document_from_retriver(vectorstore,query,no_of_documents):
    retriever = vectorstore.as_retriever(k=no_of_documents)
    docs = retriever.invoke(query)
    return docs

def init_chat_ollama(model_name):

    chat = ChatOllama(temperature=0.0, model=model_name)
    return chat


def create_chat_prompt_rag_history():
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability and the given  context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return question_answering_prompt
def create_chat_prompt_rag():
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability and the given  context:\n\n{context}",
            )
        ]
    )

    return question_answering_prompt


def create_char_message_history(history_message):

    chat_history = ChatMessageHistory()
    chat_history.add_user_message(history_message)
    return chat_history.messages


def query_rag(document_chain,chat_history,rag_docs):

    # Querying the RAG SYSTEM
    response = document_chain.invoke(
        {
            "messages":chat_history,
            "context": rag_docs,
        }
    )
    return response


def parse_retriever_input(params):
    return params["messages"][-1].content


def create_retriver_chain(parse_retriever_input,retriever,document_chain):
    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    ).assign(
        answer=document_chain,)
    return retrieval_chain



def init_postgres_db(pg_uri):

    db = SQLDatabase.from_uri(pg_uri)
    return db


def create_sql_chain(llm,db):

    chain = create_sql_query_chain(llm, db)
    return chain