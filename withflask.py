from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
import os

app = FastAPI()

# LLM and Configurations
local_llm = "zephyr-7b-beta.Q5_K_S.gguf"
config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",  # for CPU use
    **config
)

# Prompt Template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Define the route for uploading and processing the PDF file
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")

    try:
        # Load the PDF
        loader = PyPDFLoader(file.file)
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Embeddings and Vector Store Creation
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Persist directory
        persist_directory = "stores/biology_cosine"
        vector_store = Chroma.from_documents(
            texts, 
            embeddings, 
            collection_metadata={"hnsw:space": "cosine"}, 
            persist_directory=persist_directory
        )

        vector_store.persist()
        return {"message": "PDF processed and vector store created successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define the route for querying the model
@app.post("/query_model/")
async def query_model(query: str):
    try:
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
        # Load the vector store
        persist_directory = "stores/biology_cosine"
        load_vector_store = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )

        # Create the retriever
        retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})
        chain_type_kwargs = {"prompt": prompt}

        # Create the QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True, 
            chain_type_kwargs=chain_type_kwargs, 
            verbose=True
        )

        # Get the response
        response = qa(query)
        return JSONResponse(content={"response": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
