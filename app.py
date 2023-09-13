# import streamlit as st 
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplate import css, user_template, bot_template
# from langchain.llms import HuggingFaceHub

# def get_csv_text(csv_docs):
#     text = ""
#     for csv in csv_docs:
#         try:
#             text += csv.read().decode("utf-8")  # Read and decode as UTF-8
#         except Exception as e:
#             st.error(f"Error processing CSV: {str(e)}")
#             continue
#     return text

# def get_txt_text(txt_docs):
#     text = ""
#     for txt in txt_docs:
#         try:
#             text += txt.read().decode("utf-8")  # Read and decode as UTF-8
#         except Exception as e:
#             st.error(f"Error processing TXT: {str(e)}")
#             continue
#     return text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs: 
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages: 
#             text += page.extract_text()
#     return text 

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size = 1000,
#         chunk_overlap = 200,
#         length_function = len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks 

# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def get_conversation_chain(vectorstore): 
#     llm = ChatOpenAI()
#     #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1024})
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm = llm, 
#         retriever= vectorstore.as_retriever(), 
#         memory = memory
#     )
#     return conversation_chain

# # def get_conversation_chain(vectorstore): 
# #     #llm = ChatOpenAI()
# #     llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B", model_kwargs={"temperature":0.7, "max_length":500})
# #     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# #     conversation_chain = ConversationalRetrievalChain.from_llm(
# #         llm = llm, 
# #         retriever= vectorstore.as_retriever(), 
# #         memory = memory
# #     )
# #     return conversation_chain



# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
#         else: 
#             st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)

# def main():
#     load_dotenv()

#     st.set_page_config(page_title="LOG ANALYSIS AND COMPLIANCE MONITORING USING LLM's ",)

#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state or st.session_state.conversation is None:
#         # Initialize the conversation chain
#         text_chunks = ["Hello, this is an initial text chunk."]  # You can customize this
#         vectorstore = get_vectorstore(text_chunks)
#         st.session_state.conversation = get_conversation_chain(vectorstore)

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("LOG ANALYSIS AND COMPLIANCE MONITORING USING LLM's")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     st.write(user_template.replace("{{MSG}}", "Hello Chat Bot"), unsafe_allow_html=True)
#     st.write(bot_template.replace("{{MSG}}", "Hello Flipkart GRiD 5.0"), unsafe_allow_html=True)

#     with st.sidebar:
#         st.subheader("Your documents")
#         uploaded_files = st.file_uploader(
#             "Upload your documents (PDF, CSV, TXT) and click on process",
#             accept_multiple_files=True
#         )

#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 raw_text = ""
#                 pdf_docs = []
#                 csv_docs = []
#                 txt_docs = []

#                 for file in uploaded_files:
#                     if file.type == "application/pdf":
#                         pdf_docs.append(file)
#                     elif file.type == "text/csv":
#                         csv_docs.append(file)
#                     elif file.type == "text/plain":
#                         txt_docs.append(file)

#                 raw_text += get_pdf_text(pdf_docs)
#                 raw_text += get_csv_text(csv_docs)
#                 raw_text += get_txt_text(txt_docs)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)
#                 st.write(text_chunks)

#                 # create our vector store with the embeddings
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

#                 # create history

# if __name__ == '__main__':
#     main()




# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplate import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     #embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)


# if __name__ == '__main__':
#     main()


import streamlit as st 
import re
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, user_template, bot_template
from langchain.llms import HuggingFaceHub
#from main import analyze_logs

def get_csv_text(csv_docs):
    text = ""
    for csv in csv_docs:
        try:
            text += csv.read().decode("utf-8")  # Read and decode as UTF-8
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            continue
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        try:
            text += txt.read().decode("utf-8")  # Read and decode as UTF-8
        except Exception as e:
            st.error(f"Error processing TXT: {str(e)}")
            continue
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: 
            text += page.extract_text()
    return text 

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore): 
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1024})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm, 
        retriever= vectorstore.as_retriever(), 
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else: 
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="LOG ANALYSIS AND COMPLIANCE MONITORING USING LLM's ",)

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state or st.session_state.conversation is None:
        # Initialize the conversation chain
        text_chunks = ["Hello, this is an initial text chunk."]  # You can customize this
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("LOG ANALYSIS AND COMPLIANCE MONITORING USING LLM's")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Chat Bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Flipkart GRiD 5.0"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, CSV, TXT) and click on process",
            accept_multiple_files=True
        )

        
        # uploaded_log_files = st.file_uploader(
        #     "Upload your log files",
        #     accept_multiple_files=True
        # )

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = ""
                pdf_docs = []
                csv_docs = []
                txt_docs = []

                for file in uploaded_files:
                    if file.type == "application/pdf":
                        pdf_docs.append(file)
                    elif file.type == "text/csv":
                        csv_docs.append(file)
                    elif file.type == "text/plain":
                        txt_docs.append(file)
                
                # log_files = []
                # for file in uploaded_log_files:
                #     if file.type == "text/plain":
                #         log_content = file.read().decode("utf-8")
                #         csv_content = analyze_logs(log_content)
                #         st.write("Log Analysis Results:")

                #     st.download_button(
                #         label="Download Log Analysis as CSV",
                #         data=csv_content,
                #         file_name="log_analysis.csv",
                #         mime="text/csv"
                #     )

                raw_text += get_pdf_text(pdf_docs)
                raw_text += get_csv_text(csv_docs)
                raw_text += get_txt_text(txt_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create our vector store with the embeddings
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                

if __name__ == '__main__':
    main()