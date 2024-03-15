import os
import requests
from bs4 import BeautifulSoup
import json

from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import BraveSearch

import chainlit as cl

# Initialize Brave search agent with an API key and limit responses due to downstream limits
search = BraveSearch.from_api_key(
    api_key=os.environ.get("BRAVE_API_KEY"), 
    search_kwargs={"count": 3}
)

def load_brave_docs(query):
    """
    Load and parse documents from Brave search results.

    Args:
        query (str): The search query.

    Returns:
        str: A JSON string of the search result pages' contents and metadata.
    """

    # Limit query to first fifty words due to Brave search API limits
    trimmed_query = ' '.join(query.split()[:50])
    docs = json.loads(search.run(trimmed_query))

    for doc in docs:
        response = requests.get(
            doc["link"], 
            headers={'User-Agent': 'Mozilla/5.0','cache-control': 'max-age=0'}, 
            cookies={'cookies':''}
        )
        soup = BeautifulSoup(response.text, "html.parser")
        # Limit content to one thousand words due to downstream context window and token limits
        doc["text"] = ' '.join(soup.get_text(' ', strip=True).split()[:1000])
        del doc["snippet"]

    return json.dumps(docs)

@cl.on_chat_start
async def start():
    """
    Initialize user chat session by setting up the chain and session runnable.
    """

    model = ChatGroq(temperature=0.15, model_name="mixtral-8x7b-32768")
    # Prompt modified from "langchain-ai/weblangchain-response" on langchain hub
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are an expert researcher and writer, tasked with answering any question.

            Generate a comprehensive and informative, yet concise answer of 250 words or less for the given question based solely on the provided search results. You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text.

            If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.

            Anything between the following `context` html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

            <context>
                {context}
            <context/>

            REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
            """),
        ("human", "{question}")
    ])

    answer_chain = prompt | model | StrOutputParser()
    runnable = (
        { 
            "question": RunnablePassthrough(),
            "context": RunnableLambda(cl.make_async(load_brave_docs))
        }
        | RunnablePassthrough.assign(answer=answer_chain)
    )
    
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def main(message: cl.Message):
    """
    Process incoming messages and generate responses.

    Args:
        message (cl.Message): The incoming message from the user.
    """
    
    runnable = cl.user_session.get("runnable")
    response = await runnable.ainvoke(message.content)
    content = response["answer"] + "\n\n"
    context = json.loads(response["context"])

    for i, page in enumerate(context, 1):
        content += f"\n[{i}] {page['title']} ({page['link']})"

    msg = cl.Message(content=content)
    await msg.send()