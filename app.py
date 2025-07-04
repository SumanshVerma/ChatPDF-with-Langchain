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
from langchain.prompts import PromptTemplate
from html_template import css, bot_template, user_template

def get_pdf_texts(uploaded_files):
        text = ""
        for file in uploaded_files:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    
def get_chunk_texts(raw_texts, chunk_size=400):
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200, separators=["\n"], length_function=len)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4",chunk_size=chunk_size, chunk_overlap=40)
        chunks = text_splitter.split_text(raw_texts)
        return chunks

def get_vector_store(chunks):
        
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")

        embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vector_store
    
def get_conversation_history(vector_store, use_strict_prompt=False):
        llm = init_chat_model("sonar-pro", model_provider="perplexity", api_key=os.environ.get("PPLX_API_KEY"))
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        combine_docs_chain_kwargs = {}
        if use_strict_prompt:
            prompt_template = """You are a specialized assistant for answering questions based ONLY on the provided context from PDF documents. Your instructions are to be followed exactly.

            1. Review the 'Context' below.
            2. If the 'Context' contains the information to answer the 'Question', provide a helpful answer based solely on that context.
            3. If the 'Context' does NOT contain the information to answer the 'Question', you MUST respond with the exact phrase: "I can't process the request".
            4. Do NOT use any of your internal knowledge. Do NOT attempt to answer if the information is not in the 'Context'.

            Context:
            {context}

            Question: {question}
            Helpful Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            combine_docs_chain_kwargs={"prompt": PROMPT}

        conversation_chain = ConversationalRetrievalChain.from_llm (
            llm = llm , 
            retriever=vector_store.as_retriever(), 
            memory=memory,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs
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

    load_dotenv()
    # Load environment variables again to ensure they're refreshed
    load_dotenv(override=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
  
    if not os.environ.get("MISTRAL_API_KEY"):
        api_key = st.sidebar.text_input("Enter API key for MistralAI:", type="password")
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key
    
    # Add HF_TOKEN support to resolve the tokenizer warning
    if not os.environ.get("HF_TOKEN"):
        hf_token = st.sidebar.text_input("Enter Hugging Face token (optional, to resolve tokenizer warning):", type="password")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            
    # Add PPLX_API_KEY support if needed
    if not os.environ.get("PPLX_API_KEY"):
        pplx_key = st.sidebar.text_input("Enter Perplexity API key:", type="password")
        if pplx_key:
            os.environ["PPLX_API_KEY"] = pplx_key
    
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":guardsman:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs")

    if st.session_state.conversation is None and os.environ.get("MISTRAL_API_KEY"):
        # Initialize a general-purpose conversation chain to allow chatting without documents.
        try:
            embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.environ.get("MISTRAL_API_KEY"))
            # A dummy vector store is needed because ConversationalRetrievalChain requires a retriever.
            dummy_vector_store = FAISS.from_texts([" "], embedding=embeddings)
            st.session_state.conversation = get_conversation_history(dummy_vector_store)
        except Exception as e:
            st.sidebar.error(f"Failed to initialize conversation: {e}")

    with st.form(key="query_form"):
        user_question = st.text_input("Enter your query here:", key="query_input")
        submit_button = st.form_submit_button("Ask")
   
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

            st.session_state.conversation = get_conversation_history(vector_store, use_strict_prompt=True)
            
            st.success("Files processed successfully!")
        
    
if __name__ == "__main__":
    main()
