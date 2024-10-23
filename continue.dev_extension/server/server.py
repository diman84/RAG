import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

openAIEmb = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

jsLoader = GitLoader(repo_path="./", branch='master', file_filter=lambda file_path: file_path.endswith(".js"))
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=100, chunk_overlap=0
)
jsData = jsLoader.load()
js_docs = js_splitter.split_documents(jsData)
in_memory_vector_store = InMemoryVectorStore(openAIEmb)

class ContextProviderInput(BaseModel):
    query: str
    fullInput: str

app = FastAPI()


@app.post("/retrieve")
async def create_item(item: ContextProviderInput):
    docs = in_memory_vector_store.similarity_search(item.query)

    context_items = []
    for doc in docs:
        context_items.append({
            "name": doc.metadata.filename,
            "description": doc.metadata.filename,
            "content": doc.metadata.page_content,
        })

    return context_items