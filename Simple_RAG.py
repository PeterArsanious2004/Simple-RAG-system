#LLM
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values

env_values = dotenv_values("./.env")

api_key = env_values["api_key"]

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    temperature=0.8,
)

# Step 1: Load and create the knowledge base
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

def load_file(file_path: str) -> list[Document]:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    elif extension == ".docx":
        documents = Docx2txtLoader(str(path)).load()
    elif extension in {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}:
        documents = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    for document in documents:
        document.metadata.update({
            "source_type": extension.lstrip("."),
            "file_name": path.name,
            "source": str(path),
        })
    return documents

def load_knowledge_base(folder: str) -> list[Document]:
    documents = []
    for path in Path(folder).rglob("*"):
        if path.is_file():
            try:
                documents.extend(load_file(str(path)))
            except ValueError:
                print(f"Skipping unsupported file: {path}")
    return documents

documents = (load_knowledge_base("knowledge_base")
             if Path("knowledge_base").exists()
             else load_file("info.txt"))

# Chunking
from langchain_classic.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# Embedding
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

from langchain_classic.vectorstores import FAISS

vectorDB = FAISS.from_documents(chunks, embedding)
# print(f"Loaded {len(documents)} documents and created {len(chunks)} chunks.")

#Template
from langchain_core.prompts import PromptTemplate
temp = """
you are assistant, Answer the next Question using provided context,
If you don't know the answer, just say you don't know.
answer should be within 200 words or lower only
## context :
{context}
## Question :
{Question}
"""
temp = PromptTemplate.from_template(temp)

query = input("Enter your question: ")
similar_docs = vectorDB.similarity_search_with_score(query, k=4)
context = []
for document, score in similar_docs:
    source = document.metadata.get("source", "unknown source")
    context.append(f"[Source: {source}]\n{document.page_content}")

prompt = temp.format(context="\n\n".join(context), Question=query)
# print(prompt)

response = llm.invoke(prompt).content

print(f"Answer:\n{response}")

