import pandas as pd
import streamlit as st
import os
import fitz
import tempfile
from langchain.chains import RetrievalQA
import io
from streamlit_pdf_viewer import pdf_viewer
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# Set page config
st.set_page_config(page_title="📚 ChatPDF")

# Main content
st.markdown(
    "<h1 style='text-align: center;'>📚 ChatPDF</h1>", unsafe_allow_html=True
)
st.subheader("Upload a document to get started.")


# Function to load PDF from file-like object
@st.cache_data
def load_pdf(file):
    return fitz.open(stream=file, filetype="pdf")


# Custom function to extract document objects from uploaded file
def extract_documents_from_file(file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file)
    temp_file.close()

    loader = PyPDFLoader(temp_file.name)

    # Load the document
    documents = loader.load()
    return documents


def find_pages_with_excerpts(doc, excerpts):
    pages_with_excerpts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                pages_with_excerpts.append(page_num)
                break  # No need to search further on this page
    return (
        pages_with_excerpts if pages_with_excerpts else [0]
    )  # Default to the first page if no excerpts are found


@st.cache_resource
def get_llm():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPEN_AI_KEY")
    )
    return llm


@st.cache_resource
def get_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.getenv("OPEN_AI_KEY")
    )
    return embeddings


@st.cache_resource
def get_qa(_documents):
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_documents(_documents)

    # Create the vector store to use as the index
    db = Qdrant.from_documents(
        documents=texts,
        embedding=get_embeddings(),
        collection_name="my_documents",
        location=":memory:",
    )

    # Expose this index in a retriever interface
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.8}
    )

    # Create the RetrievalQA
    qa = RetrievalQA.from_chain_type(
        get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )
    return qa


def get_highlight_info(doc, excerpts):
    annotations = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                for inst in text_instances:
                    annotations.append(
                        {
                            "page": page_num + 1,
                            "x": inst.x0,
                            "y": inst.y0,
                            "width": inst.x1 - inst.x0,
                            "height": inst.y1 - inst.y0,
                            "color": "red",
                        }
                    )
    return annotations


custom_template = """
    Use the following pieces of context to answer the user question. If you
    don't know the answer, just say that you don't know, don't try to make up an
    answer.

    {context}

    Question: {question}

    Please provide your answer in the following JSON format: 
    {{
        "answer": "Your detailed answer here",
        "sources": "Direct sentences or paragraphs from the context that support 
            your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT 
            ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
    }}
    
    The JSON must be a valid json format and can be read with json.loads() in
    Python. Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_template, input_variables=["context", "question"]
)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")


if uploaded_file is not None:
    file = uploaded_file.read()

    with st.spinner("Processing file..."):
        documents = extract_documents_from_file(file)
        st.session_state.doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")

    if documents:
        qa = get_qa(documents)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! How can I assist you today? "}
            ]

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_input := st.chat_input("Your message"):
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            st.chat_message("user").write(user_input)

            with st.spinner("Generating response..."):
                try:
                    result = qa.invoke({"query": user_input})
                    parsed_result = json.loads(result['result'])

                    answer = parsed_result['answer']
                    sources = parsed_result['sources']

                    # answer = "test"
                    # sources = "Our findings indicate that the importance of science and critical thinking skills are strongly negatively associated with exposure, suggesting that occupations requiring these skills are less likely to be impacted by current LLMs. Conversely, programming and writing skills show a strong positive association with exposure, implying that occupations involving these skills are more susceptible to being influenced by LLMs."
                    sources = sources.split(". ") if pd.notna(sources) else []

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.chat_message("assistant").write(answer)

                    # Update the session state with new sources
                    st.session_state.sources = sources

                    # Set a flag to indicate chat interaction has occurred
                    st.session_state.chat_occurred = True

                except json.JSONDecodeError:
                    st.error(
                        "There was an error parsing the response. Please try again."
                    )

            # Highlight PDF excerpts
            if file and st.session_state.get("chat_occurred", False):
                doc = st.session_state.doc
                st.session_state.total_pages = len(doc)
                if "current_page" not in st.session_state:
                    st.session_state.current_page = 0

                # Find the page numbers containing the excerpts
                pages_with_excerpts = find_pages_with_excerpts(doc, sources)

                if "current_page" not in st.session_state:
                    st.session_state.current_page = pages_with_excerpts[0]

                # Save the current document state in session
                st.session_state.cleaned_sources = sources
                st.session_state.pages_with_excerpts = pages_with_excerpts

                # PDF display section
                st.markdown("### PDF Preview with Highlighted Excerpts")

                # Navigation
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if st.button("Previous Page") and st.session_state.current_page > 0:
                        st.session_state.current_page -= 1
                with col2:
                    st.write(
                        f"Page {st.session_state.current_page + 1} of {st.session_state.total_pages}"
                    )
                with col3:
                    if (
                        st.button("Next Page")
                        and st.session_state.current_page
                        < st.session_state.total_pages - 1
                    ):
                        st.session_state.current_page += 1

                # Get annotations with correct coordinates
                annotations = get_highlight_info(doc, st.session_state.sources)

                # Find the first page with excerpts
                if annotations:
                    first_page_with_excerpts = min(ann["page"] for ann in annotations)
                else:
                    first_page_with_excerpts = st.session_state.current_page + 1

                # Display the PDF viewer
                pdf_viewer(
                    file,
                    width=700,
                    height=800,
                    annotations=annotations,
                    pages_to_render=[first_page_with_excerpts],
                )
