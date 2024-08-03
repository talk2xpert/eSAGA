from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict

from rnd import LLMUTILS as LLMU


def get_llm_rag_query_results(file,query):

    MODEL_NAME = "gemma:2b"
    # 1-Load document
    loader = LLMU.load_document(file)
    # 2-Splitter
    splitted_documents = LLMU.split_document(loader)
    # 3-Create embedding of document
    vectorstore = LLMU.create_embeddings_and_store(splitted_documents, OllamaEmbeddings(model=MODEL_NAME))
    print("1",vectorstore)
    # 4-create retriver
    result_documents = LLMU.get_document_from_retriver(vectorstore=vectorstore,
                                                       query=query,
                                                       no_of_documents=2)
    print("2",result_documents)
    # 5-Initlize chat
    chat = LLMU.init_chat_ollama(MODEL_NAME)
    print("3",chat)
    # 6-create chat prompt
    question_answering_prompt = LLMU.create_chat_prompt_rag_history()
    # 7-stuffing document in llm
    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
    # 8-creating a chat message history to put in messgae history placeholder
    chat_history = LLMU.create_char_message_history("what are Objects in java?")
    # 9-querying the rag system
    response = LLMU.query_rag(document_chain, chat_history, result_documents)
   # print("4",response)
    return response
#
# results=get_llm_rag_query_results("C:\\dataset\\java.pdf" ,"what are Objects in java")
#
# print("FROM RAG")
# #result_strings = [x.dict()['page_content'] for x in results]
# print(results)









