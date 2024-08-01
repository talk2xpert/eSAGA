from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from rnd import LLMUTILS as LLMU
from langchain_community.agent_toolkits import create_sql_agent


def get_LL_DB_Response(query):
    pg_uri = f"postgresql+psycopg2://postgres:abcd@localhost:5432/books"
    db = LLMU.init_postgres_db(pg_uri)
   # print(db.dialect)
    #print(db.get_usable_table_names())
    MODEL_NAME = "gemma:2b"
    llm = LLMU.init_chat_ollama(MODEL_NAME)
    sql_chain = LLMU.create_sql_chain(llm, db)
    response = sql_chain.invoke({"question":query })
    #agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    return response

#detailed sql execution steps
def get_LL_DB_Response1():
    pg_uri = f"postgresql+psycopg2://postgres:abcd@localhost:5432/books"
    db = LLMU.init_postgres_db(pg_uri)
   # print(db.dialect)
    #print(db.get_usable_table_names())
    MODEL_NAME = "gemma:2b"
    llm = LLMU.init_chat_ollama(MODEL_NAME)

    generate_query = create_sql_query_chain(llm, db)
    query = generate_query.invoke({"question": "what is total score of Student in table student"})

    execute_query = QuerySQLDataBaseTool(db=db)
    generate_query=execute_query.invoke(query)

    chain = generate_query | execute_query

    final_answwr = chain.invoke({"question": "what is total score of `Students' from students table"})
    print(final_answwr)
    return query

get_LL_DB_Response1()

#response=get_LL_DB_Response1()
#print(response)


