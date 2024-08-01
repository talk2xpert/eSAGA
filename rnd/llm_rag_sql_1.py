from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits.spark_sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools import QuerySQLDataBaseTool, QuerySQLCheckerTool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from rnd import LLMUTILS as LLMU


#detailed sql execution steps
def get_LL_DB_Response1():

    # print(db.dialect)
    #print(db.get_usable_table_names())
    MODEL_NAME = "gemma:2b"

    # Step 1.1: Create the language model
    model = LLMU.init_chat_ollama(MODEL_NAME)

    # Step 1.2: Define connection and SQL DB tools
    pg_uri = f"postgresql+psycopg2://postgres:abcd@localhost:5432/chinook"
    db = SQLDatabase.from_uri(pg_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()
    print(tools)

    # Define the prompt with relevant settings
    prefix = SQL_PREFIX.format(dialect=toolkit.dialect, top_k=10)

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=SQL_SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        input_variables=None,
    )

    # Step 4. Define Chain
    llm_chain = LLMChain(
        llm=model,
        prompt=prompt,
        callback_manager=None,
    )

    # Step 5. Define Agent

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        prompt=prompt,
        tools=tools,
        callback_manager=None,
    )

    # Step 6. Define Agent Executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=None,
        verbose=True,
    )

    agent_executor.run("Count the customers")
# response=get_LL_DB_Response("what is total score of `Students' from students table")
# print(response)
get_LL_DB_Response1()



