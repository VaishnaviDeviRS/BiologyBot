from langchain import PromptTemplate
from langchain.llms import CTransformers
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
import gradio as gr

"""
This script initializes a Zephyr 7B Beta LLM model for a Retrieval-Augmented Generation (RAG) task, 
using a custom prompt template and integrating with a vector store for biology-related content. 
The script is designed to take user input and provide answers based on the given context.

Initialization:
---------------
- The script begins by setting up the local LLM model and its configuration parameters.
- A prompt template is defined, guiding how the LLM should respond to user queries.
- A pre-trained embedding model is used to manage the text-to-vector transformation.
- A vector store containing biology content is loaded for retrieving relevant context during queries.

Functions and Components:
--------------------------
- **local_llm**: Specifies the local model file used for the LLM.
- **config**: A dictionary containing settings for the LLM, such as token generation limits, temperature, and threading options.
- **llm**: Initializes the LLM model with the specified configuration using the CTransformers library.
- **prompt_template**: Defines the structure and rules for generating responses from the LLM, including how to use the context and question.
- **model_name, model_kwargs, encode_kwargs**: Parameters for the HuggingFaceBge Embeddings model that converts text into vector embeddings.
- **embeddings**: The embedding function initialized with the specified model and settings.
- **load_vector_store**: Loads the Chroma vector store with precomputed embeddings for biology-related content.
- **retriever**: A retriever object for fetching relevant documents from the vector store based on the user's query.
- **get_response**: A function that handles the user query, retrieves relevant documents, and generates a response using the LLM.
- **iface**: A Gradio interface that provides a web-based UI for interacting with the model. It accepts user input, runs the get_response function, and displays the output.

Example Usage:
--------------
- The script includes sample prompts such as "What is Carbohydrate?" and "What is Chemical Bond?".
- Users can interact with the bot through a simple web interface, inputting questions related to biology and receiving contextual answers.

Deployment:
-----------
- The script can be run locally, with Gradio launching a web interface for user interaction.
- Debugging options are available through Gradio's `launch()` function, where errors can be inspected by setting `debug=True`.
"""


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
    lib="avx2", #for CPU use
    **config
)

print("LLM Initialized")


prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/Biology_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})


print("Doc Retrieved")

chain_type_kwargs = {"prompt": prompt}

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents = True,
#     chain_type_kwargs= chain_type_kwargs,
#     verbose=True
# )

# response = qa(query)

# print(response)

sample_prompts = ["what is Carbohyrate?", "What is Chemical Bond?"]

def get_response(input):
  query = input
  chain_type_kwargs = {"prompt": prompt}
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
  response = qa(query)   #qa.invoke(query) if deprecated warning or error is thrown
  return response

input = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

iface = gr.Interface(fn=get_response, 
             inputs=input, 
             outputs="text",
             title="My Biology Bot",
             description="This is a RAG implementation based on Zephyr 7B Beta LLM.",
             examples=sample_prompts,
             allow_flagging=False
             )

iface.launch()  # set debug=True inside launch to check for errors if any










            







