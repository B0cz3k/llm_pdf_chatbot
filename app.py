import os
import torch
import streamlit as st
from model import ChatModel
from utils import Encoder, FaissDB, loader_splitter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

st.title("LLM RAG Chatbot")

@st.cache_resource
def load_model(model="google/gemma-2b-it"):
    return ChatModel(model_id=model, device=DEVICE)

@st.cache_resource
def load_encoder(model="sentence-transformers/all-MiniLM-L12-v2"):
    return Encoder(model_name=model, model_kwargs={'device': DEVICE})

def save_file(file):
    file_path = os.path.join(FILES_DIR, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    return file_path

model = load_model()
encoder = load_encoder()

with st.sidebar:
    # inputs and parameters in the sidebar
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 2048)
    k = st.number_input("k", 1, 10, 6)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
    )
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(save_file(uploaded_file))
    if uploaded_files != []:
        # create vector database from retrieved documents
        docs = loader_splitter(file_paths)
        DB = FaissDB(docs=docs, embedding_function=encoder.embedding_function)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        context = (
            None if uploaded_files == [] else DB.similarity_search(user_prompt, k=k)
        )
        answer = model.generate(
            user_prompt, context=context, max_new_tokens=max_new_tokens
        )
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})