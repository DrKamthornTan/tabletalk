import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from apikey import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title('CSV Question and answer ChatBot')

csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")

if csv_file_uploaded is not None:
    def save_file_to_folder(uploadedFile):
        # Save uploaded file to 'content' folder.
        save_folder = 'content'
        save_path = Path(save_folder, uploadedFile.name)
        with open(save_path, mode='wb') as w:
            w.write(uploadedFile.getvalue())

        if save_path.exists():
            st.success(f'File {uploadedFile.name} is successfully saved!')
            
    save_file_to_folder(csv_file_uploaded)
    
    loader = CSVLoader(file_path=os.path.join('content/', csv_file_uploaded.name))

    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")