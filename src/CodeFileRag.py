import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import gradio as gr

PROJECT_PATH = "C:\\Users\\SMehi\\source\\repos"
FAISS_FOLDER = "C:\\ProgramData\\test"

import os
import fnmatch
from langchain.schema import Document as LangchainDocument  # adjust if needed


def load_code_files_from_folder(folder_path, extensions=(".cs",".xml",".json"), exclude_patterns=None):
    if exclude_patterns is None:
        exclude_patterns = []

    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(extensions):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, folder_path)

            # Skip if rel_path matches any exclude pattern
            if any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns):
                #print(f"Skipped {full_path}: matches an exclude pattern")
                continue

            try:
                print(f"Processing file {full_path}...")
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                    docs.append(LangchainDocument(
                        page_content=code,
                        metadata={"source": rel_path}
                    ))
                print(f"DONE processing file {full_path}...")
            except Exception as e:
                print(f"Skipped {full_path}: {e}")
    return docs


exclude_paths = [
    "*/.vs/*",
    "*/bin/*",
    "*/obj/*",
    "*/References/*",
    "*/development_env/*"]

# Load documents
documents = load_code_files_from_folder(PROJECT_PATH, exclude_patterns=exclude_paths)

# Smart chunking for code
# Chunk code with function-aware separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\npublic ",    # C# methods / properties
        "\nprivate ",   # C# internal methods
        "\nprotected ", # C# methods
        "\nclass ",     # C#, Python
        "\nvoid ",      # C# functions
        "\nstatic ",    # C#
        "\ndef ",       # Python
        "\n<",          # XML tag start
        "\n{",          # JSON / C# block start
        "\n\n",         # general paragraph/logical block break
        "\n",           # line-based fallback
        " ",            # word-based fallback
        ""              # character fallback
    ]
)
chunks = text_splitter.split_documents(documents)

embedding_model = "BAAI/bge-small-en-v1.5"
# Use BGE Code Embeddings (simple + effective)
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

if os.path.exists(os.path.join(FAISS_FOLDER, "index.faiss")):
    print(f"Loading FAISS index from {FAISS_FOLDER}...")
    vectorstore = FAISS.load_local(FAISS_FOLDER, embedding_model)
else:
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_FOLDER)
    print(f"Saved FAISS index to {FAISS_FOLDER}")
    vectorstore = vectorstore

# Build FAISS vector store
#vectorstore = FAISS.from_documents(chunks, embeddings)

#Connect to local LLM via Ollama
llm = OllamaLLM(model="codellama:13b")  # or mistral, gemma, etc.


# RetrievalQA pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Gradio UI
def answer_query(query):
    response = qa({"query": query})
    result = response["result"]
    sources = [doc.metadata["source"] for doc in response["source_documents"]]
    sources_text = "\n".join(set(sources))
    return f"Answer:\n{result}\n\n Sources:\n{sources_text}"

gr.Interface(fn=answer_query, inputs="text", outputs="text", title="Local Code RAG").launch()
