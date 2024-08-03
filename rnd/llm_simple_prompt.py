#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langserve import add_routes
from langchain_community.document_loaders import WebBaseLoader

# 1. Create prompt template
# Example 1
system_template = "Translate the following into {language}:"
prompt_template1 = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])
# Example 2
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        ('user', '{text}')
            ]
)

# 2. Create model
model_name = "gemma:2b"
model = ChatOllama(temperature=0.0, model=model_name)
# 3. Create parser
parser = StrOutputParser()

# # 4. Create chain
chain = prompt_template | model | parser

#1
#response =chain.invoke({"language": "french", "text": "hello"})
#2
#response =chain.invoke({"text":"what are Objects in java"})
#print(response)

