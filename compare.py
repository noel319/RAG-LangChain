from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import warnings
import os
import logging
warnings.filterwarnings("ignore")

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
logging.basicConfig(filename="conversation_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")

openai_api_key = os.environ["OPENAI_API_KEY"]

def response(question):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    original_output = completion.choices[0].message.content

    return original_output

data = []
txt_loader = DirectoryLoader(
        "txtdata", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
data.extend(txt_loader.load())
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(data)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory="db"
)
def reformat_query(query):
    reformulation_prompt = f"Reformulate the query for better document retrieval: {query}"
    return response(reformulation_prompt)  # Using OpenAI to reformat the query

def conversation_chat(query, chain, history):
    result = chain({
        "question": query,
        "chat_history": history
    })
    history.append((query, result["answer"]))
    logging.info(f"User Query: {query}\nRAG Answer: {result['answer']}")
    return result["answer"]

def combined_response(query, rag_response):
    original_output = response(query)
    combined = f"**RAG ChatGPT:** {rag_response}\n\n**OpenAI ChatGPT:** {original_output}"    
    return combined

def handle_low_confidence(rag_response):
    if not rag_response or "I'm not sure" in rag_response:  # Simple condition, can be expanded
        return "I couldn't find enough information on that. Can you clarify or rephrase the question?"
    return rag_response

def create_conversational_chain(vector_store):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=openai_api_key
    )

    memory = ConversationSummaryMemory(
        llm = llm,
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    return chain
vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory="db"
        )

history = []
chain = create_conversational_chain(vector_store=vectorstore)



while True:
    query = input()
    if query == "exit":
        break
    elif query == "reset":
        history=[]
        continue
    refined_query = reformat_query(query)
    # Get RAG response
    rag_response = conversation_chat(query=refined_query, chain=chain, history=history)
    
    # Handle low-confidence responses
    rag_response = handle_low_confidence(rag_response)
    # Combine RAG and OpenAI responses
    final_response = combined_response(query, rag_response)
    print(final_response)
