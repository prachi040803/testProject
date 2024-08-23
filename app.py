import os
import pickle
import streamlit as st
import PyPDF2
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Open-source embeddings alternative
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from load_dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

GROQ_API_KEY="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"
PINECONE_API_KEY="68102030-4d68-47fb-b3fe-92f63ded2015"
COHERE_API_KEY="FNvFsoaegBiM8Etyqm1BwLH7BzOkOCNRsLR4rO4k"

# Initialize LLM Groq
llm_groq = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name='llama3:8b'
)

# Load preprocessed data if exists, otherwise preprocess and save data
def load_or_preprocess_data(pdf_folder_path):
    if os.path.exists("all_texts.pkl") and os.path.exists("embeddings.pkl"):
        with open("all_texts.pkl", "rb") as f:
            all_texts = pickle.load(f)
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return all_texts, embeddings

    # Initialize an empty list to hold all the texts
    all_texts = []

    # Function to process individual PDF files
    def process_pdf(file_path):
        pdf = PyPDF2.PdfReader(file_path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
        return pdf_text

    # Process all PDFs in parallel
    with ThreadPoolExecutor() as executor:
        pdf_texts = list(executor.map(process_pdf, [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]))
    
    for pdf_text in pdf_texts:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(pdf_text)
        all_texts.extend(texts)
    
    # Use Sentence Transformers for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and efficient model
    embeddings = model.encode(all_texts, convert_to_tensor=True)

    # Save processed data for future use
    with open("all_texts.pkl", "wb") as f:
        pickle.dump(all_texts, f)
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    return all_texts, embeddings

def process_query(query):
    try:
        pdf_folder_path = "/content/drive/MyDrive/Capstone_data/"

        # Load or preprocess data
        all_texts, embeddings = load_or_preprocess_data(pdf_folder_path)

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "thapargenie2"

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=embeddings.shape[1],  # Set the correct dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        index = pc.Index(index_name)

        # Upsert data into Pinecone
        for i, embedding in enumerate(embeddings):
            index.upsert([((str(i), embedding.numpy().tolist(), {"text": all_texts[i]}))])

        # Generate embedding for the query
        model = SentenceTransformer('all-MiniLM-L6-v2')
        question_embedding = model.encode(query, convert_to_tensor=True)

        # Query the index
        query_result = index.query(vector=question_embedding.numpy().tolist(), top_k=5, include_metadata=True)

        # Extract metadata from query result
        docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}

        # Rerank the documents using Cohere
        co = cohere.Client(COHERE_API_KEY)
        rerank_docs = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=list(docs.keys()),
            top_n=5,
            return_documents=True
        )

        # Extract reranked documents
        reranked_texts = [doc.document.text for doc in rerank_docs.results]

        # Combine the reranked texts into context
        context = " ".join(reranked_texts)

        # Create a template for the final response
        Template = f"Based on the following context: {context} generate a precise summary related to the question: {query}. Do not remove necessary information related to context. Consider `\n` as newline character."
        filled_template = Template.format(context=context, question=query)

        # Generate the final response using Groq
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": filled_template,
                }
            ],
            model="llama3-70b-8192",
        )

        # Return the final response
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit interface
st.title("ThaparGenie - An LLM Based Chatbot")
st.write("Ask your query related to specialization courses in Civil Engineering.")

query = st.text_input("Enter your query here:")
if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            response = process_query(query)
            st.write(response)
    else:
        st.write("Please enter a query.")

