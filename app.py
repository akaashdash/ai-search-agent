from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

@cl.on_chat_start
async def start():
    model = ChatGroq(temperature=0.25, model_name="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant."
            ),
            ("human", "{question}")
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        { "question" : message.content },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    ):
        await msg.stream_token(chunk)

    await msg.send()