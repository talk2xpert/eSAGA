from langchain_community.embeddings import OllamaEmbeddings

import rnd.LLMUTILS as LLMU


def get_Vector_DB_Results(pdf_file,query):
    MODEL_NAME = "gemma:2b"
    # 1-Load document
    loader = LLMU.load_document(pdf_file)
    # 2-Splitter
    splitted_documents = LLMU.split_document(loader)
    print(splitted_documents)
    # 3-Create embedding of document
    vectorstore = LLMU.create_embeddings_and_store(splitted_documents, OllamaEmbeddings(model=MODEL_NAME))
    # 4-search from vector_db_similarity
    results = LLMU.get_document_from_vector_db_embedding(vectorstore,query,1)
    return results


# results=get_Vector_DB_Results("C:\\dataset\\java.pdf" ,"what are Objects in java")
#
# print("FROM VECOTR EMBEDDINGS")
# result_strings = [x.dict()['page_content'] for x in results]
# print(result_strings)





