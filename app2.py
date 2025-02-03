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
from langchain_core.messages import HumanMessage, SystemMessage
os.environ["GOOGLE_API_KEY"] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"
# --- Streamlit Page Config ---
st.set_page_config(page_title="🤖 Smarte-KI Chatbot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
        body { background-color: #2D2D2D; color: white; font-family: 'Arial', sans-serif; }
        .chat-header { text-align: center; font-size: 26px; font-weight: bold; padding: 15px;
            background: linear-gradient(90deg, #007BFF, #00D4FF); border-radius: 10px; color: white;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); }
        .chat-container { display: flex; flex-direction: column; align-items: flex-start;
            width: 100%; max-width: 800px; margin: auto; }
        .question-container, .response-container { display: flex; margin-bottom: 15px; }
        .question-container { justify-content: flex-end; }
        .response-container { justify-content: flex-start; }
        .question, .response { max-width: 75%; padding: 15px; border-radius: 15px; font-size: 16px;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2); }
        .question { background-color: #007BFF; color: white; text-align: right; }
        .response { background-color: #00D4FF; color: black; }
        .stTextInput > div > div { border-radius: 10px !important; border: 2px solid #00D4FF !important; }
        .emoji { font-size: 20px; margin-right: 10px; }
    </style>
    <div class='chat-header'>🤖 Smarte-KI.de Chatbot</div>
    <p style='text-align: center;'>Welcome! Ask me anything about Smarte-KI’s AI solutions.</p>
    """,
    unsafe_allow_html=True
)

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input Box ---
user_query = st.chat_input("💬 Type your question:")

# --- Load Vectorstores ---
@st.cache_resource
def load_vectorstore():
    # Define paths for the two vector stores
    vectorstore_path1 = "data_vectorstore"  # For books
    vectorstore_path2 = "faq_vectorstore"   # For FAQs
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load or create the vector store for books
    if os.path.exists(vectorstore_path1):
        vectorstore_books = FAISS.load_local(vectorstore_path1, embeddings, allow_dangerous_deserialization=True)
        print("Books vectorstore loaded from local storage.")
    else:
        print("Processing books data...")
        book_file_paths = [
            "data/01_book.pdf", 
            "data/02_book.pdf"
        ]
        
        docs_books = []
        for path in book_file_paths:
            loader = PyPDFLoader(path)
            docs_books.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks_books = text_splitter.split_documents(docs_books)

        vectorstore_books = FAISS.from_documents(chunks_books, embeddings)
        vectorstore_books.save_local(vectorstore_path1)
        print("Books vectorstore created and saved locally.")

    # Load or create the vector store for FAQs
    if os.path.exists(vectorstore_path2):
        vectorstore_faq = FAISS.load_local(vectorstore_path2, embeddings, allow_dangerous_deserialization=True)
        print("FAQs vectorstore loaded from local storage.")
    else:
        print("Processing FAQ data...")
        faq_file_paths = [
            "data/use_Case.pdf", 
            "data/web_data.pdf"
        ]
        
        docs_faq = []
        for path in faq_file_paths:
            loader = PyPDFLoader(path)
            docs_faq.extend(loader.load())

        chunks_faq = text_splitter.split_documents(docs_faq)

        vectorstore_faq = FAISS.from_documents(chunks_faq, embeddings)
        vectorstore_faq.save_local(vectorstore_path2)
        print("FAQs vectorstore created and saved locally.")

    return vectorstore_books, vectorstore_faq

vectorstore_books, vectorstore_faq = load_vectorstore()
retriever_books = vectorstore_books.as_retriever(search_kwargs={"k": 3})
retriever_faq = vectorstore_faq.as_retriever(search_kwargs={"k": 3})

# --- Initialize Chat Model ---
@st.cache_resource
def init_chat():
    return ChatGroq(
                    temperature=0.1,
                    groq_api_key="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",
                    model_name="llama3-70b-8192"
                    )

chat = init_chat()

# --- Prompt Template ---
prompt_template = """
You are an **AI sales representative** for **Smarte-KI.de** 🤖💡, specializing in selling **AI solutions** 🚀.
Your job is to give **convincing, engaging and concise** answers

---

### **🔥 Smarte-KI’s AI solutions:**
1️⃣ **Production & Electronics** ⚙️🔧 → Predictive maintenance, fault detection.
2️⃣ **Computer Vision** 📸🎯 → Inventory tracking, automation, quality control.
3️⃣ **NLP (speech processing)** 🗣️🤖 → AI chatbots, knowledge management, supplier automation.
4️⃣ **Data analysis** 📊🔍 → Demand forecasting, process optimization.
5️⃣ **Healthcare** 🏥🧑‍⚕️ → AI for medical imaging, patient analytics.
6️⃣ **Logistics** 🚚📦 → Fleet optimization, route planning, warehouse management with AI.
7️⃣ **Agriculture & Energy** 🌱⚡ → Crop monitoring, AI for renewable energy.
8️⃣ **Real Estate & Construction** 🏗️🏢 → Smart planning, AI-based assessments.

---

### **🎯 Response Guidelines (Short, Effective & Attractive)**
✅ Keep responses to **maximum 2-4 lines** unless the user requests more details.
✅ **Use relevant emojis** for engaging, visual communication.
✅ Be **clear, direct and convincing** - avoid complicated explanations.
✅ **Emphasize the business benefit** → increase sales, efficiency, productivity.
✅ If a user needs a detailed explanation, provide a **structured but engaging** answer.
✅ **If a user wants to know more, we ask them for their Gmail address and phone number to arrange a meeting.""

---

### **🔹 Context:**
{context}

### **🔹 User question:**
{question}



### **🔹 Answer:**
You are the **AI sales chatbot** of Smarte-KI.de 🤖.
Answer **Your answer should be very concise and short. Please make sure that your answer is very short and concise as well as correct.
Use **clear language, strong sales arguments & appropriate emojis** for an appealing response!"""

prompt = ChatPromptTemplate.from_template(prompt_template)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Process User Input ---
if user_query:
    with st.spinner("💡 Generating response..."):
        chat_history = [HumanMessage(content=user_query)]

        # Directly use FAISS vectorstore for similarity search with scores
        retrieved_books_with_scores = vectorstore_books.similarity_search_with_score(user_query, k=5)
        retrieved_faq_with_scores = vectorstore_faq.similarity_search_with_score(user_query, k=5)

        # Extract documents and similarity scores separately
        retrieved_books = [doc[0] for doc in retrieved_books_with_scores] 
        retrieved_faq = [doc[0] for doc in retrieved_faq_with_scores] 

        scores_books = [doc[1] for doc in retrieved_books_with_scores] 
        scores_faq = [doc[1] for doc in retrieved_faq_with_scores]  

     
        avg_score_books = sum(scores_books) / max(len(scores_books), 1) if scores_books else 0.0
        avg_score_faq = sum(scores_faq) / max(len(scores_faq), 1) if scores_faq else 0.0

    
        # st.write(f"Retrieved Books: {retrieved_books}")
        # st.write(f"Retrieved FAQs: {retrieved_faq}")
        # st.write(f"📖 Avg Similarity Score (Books): {avg_score_books}")
        # st.write(f"📜 Avg Similarity Score (FAQs): {avg_score_faq}")

        
        if not retrieved_books and not retrieved_faq:
            response = "❌ I couldn't find relevant information. Would you like to book a demo with Smarte-KI?"
            source = "🤷 No relevant source found."
        else:
            # Select the best retriever based on similarity score
            if avg_score_books > avg_score_faq:
                selected_retriever = retriever_books
                source = "📖 Books"
            else:
                selected_retriever = retriever_faq
                source = "📜 FAQs"

            # Create RetrievalQA Chain with the selected retriever
            chain = RetrievalQA.from_chain_type(
                llm=chat,
                chain_type="stuff",
                retriever=selected_retriever,  # CORRECT retriever
                memory=memory,
                chain_type_kwargs={"prompt": prompt}
            )

            
            result = chain.invoke({"query": user_query,"chat_history": chat_history})
            response = result if isinstance(result, str) else result.get("result", "")

        # Debugging - Show AI response
        # st.write("🤖 AI Response:", response)

        # Append to chat history
        st.session_state.chat_history.append({
            "query": user_query,
            "response": response,
            "source": source
        })

# --- Display Chat History ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for entry in st.session_state.chat_history:
    st.markdown(f"<div class='question-container'><div class='emoji'>💬</div><div class='question'>{entry['query']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='response-container'><div class='emoji'>🤖</div><div class='response'>{entry['response']} <br><small></small></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
