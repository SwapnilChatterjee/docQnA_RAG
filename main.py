import os
import streamlit as st
import shutil
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vectorsoredb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings ##vector embedding

from dotenv import load_dotenv

custom_env_path = "/Users/swapnil/Desktop/projects/pdfQnAChatBot/configs/.env"
load_dotenv(custom_env_path)

## loading the GROQ API KEY and GOGGLE generative AI embeddings
groq_api_key = os.getenv("GROQ_DOC_QnA_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# print(groq_api_key)

st.title("Document QnA ChatBot using Gemma Model")

## loading the llm of gemma model
llm = ChatGroq(api_key=groq_api_key, model="Gemma2-9b-It")

##setting up prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state or "vectors" in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") ##DAta Ingestion
        st.session_state.docs = st.session_state.loader.load() ##Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

if "file_uploader" not in st.session_state:
    st.session_state.file_uploader = True
if st.session_state.file_uploader == True:
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "pdf", "csv", "txt", "xlsx"])

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the local directory."""
    file_path = os.path.join(os.getenv('UPLOAD_DIRECTORY'), uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def delete_old_files():
    for item in os.listdir(os.getenv('UPLOAD_DIRECTORY')):
        item_path = os.path.join(os.getenv('UPLOAD_DIRECTORY'), item)
        
        # Check if the item is a file or directory
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted directory: {item_path}")

if uploaded_file is not None:
    # Save the uploaded file
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"File uploaded successfully!")

if st.button("Delete old memory"):
    # if "embeddings" in st.session_state:
    #     st.session_state.embeddings = ""
    # if "loader" in st.session_state:
    #     st.session_state.loader = ""
    # if "docs" in st.session_state:
    #     st.session_state.docs = ""
    # if "splitter" in st.session_state:
    #     st.session_state.text_splitter = ""
    # if "final_documents" in st.session_state:
    #     st.session_state.final_documents = ""
    if "vectors" in st.session_state:
        st.session_state.vectors = ""
    if "vector_store_db" in st.session_state:
        st.session_state.vector_store_db = False
    if "document_uploaded" in st.session_state:
        st.session_state.document_uploaded = False
    if "ask_question" in st.session_state:
        st.session_state.ask_question = False
    if "file_uploader"  in st.session_state:
        st.session_state.file_uploader = True
    if "learn_more" in st.session_state:
        st.session_state.learn_more = False
    delete_old_files()

if "vector_store_db" not in st.session_state:
    st.session_state.vector_store_db = False
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "ask_question" not in st.session_state:
    st.session_state.ask_question = False
if "learn_more" not in st.session_state:
    st.session_state.learn_more = False
## A button is created once the user clicks it the vector embeddings of the chosen documents are stored and ready for query

    

if st.session_state.document_uploaded == False:
    if st.button("Documents uploaded"):
        st.write("Document Uploaded Successfully")
        st.session_state.document_uploaded = True

# if st.session_state.learn_more == True :       
#     st.session_state.vector_store_db = False
if st.session_state.document_uploaded == True and st.session_state.vector_store_db == False:
    if st.button("Learn the documents"):
        # st.write(st.session_state.vector_store_db)
        vector_embedding()
        st.write("Vector store DB is ready")
        st.session_state.vector_store_db = True
        st.session_state.learn_more = True

if st.session_state.learn_more == True:
    # st.session_state.vector_store_db = False
    if st.button("Learn More"):
        st.session_state.vector_store_db = False
        st.session_state.learn_more = False

if st.session_state.ask_question == False and st.session_state.vector_store_db == True:
    ## Showing the initial comment in the UI
    prompt1 = st.text_input("WHAT DO YOU WANT TO ASK FROM THE DOCUMENTS")
    
    import time

    try:
        if prompt1:
            
            document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
            
            start = time.process_time()
            st.write(f"Answer generation time taken: {start}")
            response = retrieval_chain.invoke({'input':prompt1})

            st.write(f"Question: {prompt1}")
            st.write(f"Answer: {response['answer']}")

            #gemma model along with giving the answer it also gives some context so we will display the response context from where it is giving the asnwer
            with st.expander("Relatable content from the documents"):
                #Find the relavant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("----------------------")
    except Exception as e:
        st.write("I am still a kid give me docs to learn!!!!")
        AssertionError(f"The error : {e}")







