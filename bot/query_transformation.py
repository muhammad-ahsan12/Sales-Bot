import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Set environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"
os.environ['GROQ_API_KEY'] = "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"

# Streamlit configuration
st.set_page_config(page_title="🤖 Smarte-KI Chatbot", page_icon="🤖", layout="centered")

# --- Custom Styling (Dark Theme, Chat Bubbles, Modern Look) ---
st.markdown(
    """
    <style>
        body {
            background-color: #2D2D2D;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .chat-header {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            padding: 15px;
            background: linear-gradient(90deg, #007BFF, #00D4FF);
            border-radius: 10px;
            color: white;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            width: 100%;
            max-width: 800px;
            margin: auto;
        }
        .question-container, .response-container {
            display: flex;
            margin-bottom: 15px;
        }
        .question-container {
            justify-content: flex-end;
        }
        .response-container {
            justify-content: flex-start;
        }
        .question, .response {
            max-width: 75%;
            padding: 15px;
            border-radius: 15px;
            font-size: 16px;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2);
        }
        .question {
            background-color: #007BFF;
            color: white;
            text-align: right;
        }
        .response {
            background-color: #00D4FF;
            color: black;
        }
        .stTextInput > div > div {
            border-radius: 10px !important;
            border: 2px solid #00D4FF !important;
        }
    </style>
    <div class='chat-header'>🤖 Smarte-KI.de Chatbot</div>
    <p style='text-align: center;'>Welcome! Ask me anything about Smarte-KI.de’s AI solutions.</p>
    """,
    unsafe_allow_html=True
)

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Generate Query Variations for Better Retrieval
prompt_template = """
You are an AI language model assistant. Generate five variations of the given user question to enhance document retrieval.
Original question: {question}
"""
prompt_perspectives = ChatPromptTemplate.from_template(prompt_template)

def generate_queries(user_query):
    chain = (
        prompt_perspectives
        | ChatGroq(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    return chain.invoke({"question": user_query})

# Load Vectorstore Function
@st.cache_resource
def load_vectorstore():
    vectorstore_path = "new_local_vectorstore"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        print("Vectorstore loaded from local storage.")
    else:
        print("Processing documents...")
        file_paths = [
            "new_Data/ger_to_eng 02.pdf", 
            "new_Data/ger_to_eng.pdf", 
            "new_Data/use_Case.pdf", 
            "new_Data/website_data.pdf"
        ]
        
        docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vectorstore_path)
        print("Vectorstore created and saved locally.")

    return vectorstore

vectorstore = load_vectorstore()

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# Define the prompt for the chatbot
prompt_template = """
You are an AI-powered **Sales & Support Chatbot** for **Smarte-KI.de**, a company specializing in advanced AI solutions. Your goal is to provide **concise, informative, and accurate** responses about Smarte-KI.de’s AI products, services, and applications across industries.

--- 

### **🚀 How You Should Respond:**
✅ **Never start with a greeting** (e.g., "Hello" or "Hi") unless explicitly asked.  
✅ **Keep responses between 3 to 8 lines** for clarity and engagement.  
✅ **Always provide correct and concise responses** without unnecessary details.  
✅ **Never mention a "vector database" or internal system limitations.**  
✅ **Focus only on Smarte-KI.de’s AI solutions and use cases.**  

--- 

### **🔹 AI Solutions & Use Cases You Support:**
1️⃣ **Manufacturing & Electronics** → Predictive maintenance, defect detection, quality control.  
2️⃣ **Computer Vision** → Real-time inventory tracking, automation, and object recognition.  
3️⃣ **Natural Language Processing (NLP)** → AI chatbots, knowledge management, and supplier automation.  
4️⃣ **Data Analytics** → Demand forecasting, predictive insights, and process optimization.  
5️⃣ **Healthcare** → AI-powered medical imaging, patient analytics, and clinical automation.  
6️⃣ **Logistics & Transportation** → AI for fleet optimization, route planning, and warehouse management.  
7️⃣ **Agriculture & Energy** → AI-driven crop monitoring, renewable energy forecasting, and infrastructure analysis.  
8️⃣ **Real Estate & Construction** → Smart planning, AI-powered property valuation, and design optimization.  

--- 

### **📌 Response Rules:**
✅ **Company Identity:** If asked, clearly state:  
   *"Smarte-KI.de is a leading provider of AI solutions, offering automation, predictive maintenance, and NLP-powered assistants to optimize business operations."*  

✅ **No Greetings in Responses:** Avoid unnecessary openings like "Hello," "Sure," or "I can help with that."  
✅ **Keep It Short & Professional:** Responses should be **clear, engaging, and within 3 to 8 lines.**  
✅ **Objection Handling (Sales Strategy):**  
   - **Price Concern:**  
     *"Our AI solutions reduce long-term costs by optimizing efficiency and minimizing downtime. Would you like a breakdown of ROI?"*  
   - **Hesitation:**  
     *"What specific concerns do you have? I can provide case studies or additional details to help you decide."*  
   - **Competitor Inquiry:**  
     *"Smarte-KI.de solutions are designed for high accuracy, seamless integration, and scalability. Would you like a feature comparison?"*  

✅ **No Speculative or Off-Topic Answers:**  
   - **Wrong:** "I'm sorry, I don't have that information."  
   - **Right:** "I provide AI solutions for Smarte-KI.de. Let me know what you're looking for in AI automation or analytics."  

✅ **Use Context for Personalization:**  
   - If the user previously asked about **predictive maintenance**, reference it in follow-up answers.  
   - If the user mentioned **budget concerns**, reinforce AI's long-term cost benefits.  

--- 

### **🔹 Context:**  
{context}  

### **🔹 Question:**  
{question}  

### **🔹 Response:**  
Provide a **precise and structured answer (3-8 lines)** without greetings. Ensure clarity, accuracy, and alignment with Smarte-KI.de’s AI offerings.  
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# User Input Box
user_query = st.chat_input("💬 Type your question:")

# If user input is received
if user_query:
    # Step 1: Generate query variations for better retrieval
    retrieval_chain = generate_queries | vectorstore.as_retriever().map() | get_unique_union 
    docs = retrieval_chain.invoke({"question": user_query})
    
    # Step 4: Generate response based on combined context
    final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | ChatGroq(temperature=0) 
    | StrOutputParser()
)

    response = final_rag_chain.invoke({"question": user_query})
    
    # Step 5: Display response in chat
    st.session_state.chat_history.append({"question": user_query, "response": response})
    
    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        st.markdown(f"<div class='question-container'><div class='emoji'>💬</div><div class='question'>{entry['question']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='response-container'><div class='emoji'>🤖</div><div class='response'>{entry['response']}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
