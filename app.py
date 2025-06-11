import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS    
from langchain_mistralai import MistralAIEmbeddings
import os
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from html_template import css, bot_template, user_template

def get_pdf_texts(uploaded_files):
        text = ""
        for file in uploaded_files:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    
def get_chunk_texts(raw_texts, chunk_size=1000):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200, separators=["\n"], length_function=len)
        chunks = text_splitter.split_text(raw_texts)
        return chunks

def get_vector_store(chunks):
        
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")

        embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vector_store
    
def get_conversation_history(vector_store):
        llm = init_chat_model("sonar-pro", model_provider="perplexity", api_key=os.environ.get("PPLX_API_KEY"))
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm (
            llm = llm , 
            retriever=vector_store.as_retriever(), 
            memory=memory
        )
        return conversation_chain
def handle_user_input(user_question):
     response = st.session_state.conversation({"question": user_question})
     st.session_state.chat_history = response['chat_history']

     for i, message in enumerate(st.session_state.chat_history):
        if i % 2 != 0:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
     
    

def main():
    # Load environment variables
    load_dotenv()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
  
    if not os.environ.get("MISTRAL_API_KEY"):
        api_key = st.sidebar.text_input("Enter API key for MistralAI:", type="password")
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key
            
    # Set up the Streamlit app  
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":guardsman:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs")
    
    # Create a form to prevent auto-submission
    with st.form(key="query_form"):
        user_question = st.text_input("Enter your query here:", key="query_input")
        submit_button = st.form_submit_button("Ask")
        
    # Only process when form is submitted
    if submit_button and user_question:
        handle_user_input(user_question)
  


    
    with st.sidebar:
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            for file in uploaded_files:
                st.write(f"Uploaded Successfully")
        if st.button("Process the files", key="send_button"):
          with st.spinner("Processing..."):
            raw_texts = get_pdf_texts(uploaded_files)
            
            chunks = get_chunk_texts(raw_texts)


            vector_store = get_vector_store(chunks)

            st.session_state.conversation = get_conversation_history(vector_store)


            # st.write(chunks)
            st.success("Files processed successfully!")
        
    
    
    # Add more functionality here as needed
if __name__ == "__main__":
    main()
