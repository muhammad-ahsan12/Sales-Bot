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

# --- Custom Styling (Dark Theme, Chat Bubbles, Modern Look) ---
st.markdown(
    """
    <style>
        /* Background & Font */
        body {
            background-color: #2D2D2D;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        
        /* Chatbot Header */
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

        /* Chat Containers */
        .chat-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            width: 100%;
            max-width: 800px;
            margin: auto;
        }

        /* Chat Bubbles */
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

        /* Input Box */
        .stTextInput > div > div {
            border-radius: 10px !important;
            border: 2px solid #00D4FF !important;
        }

        /* Emoji */
        .emoji {
            font-size: 20px;
            margin-right: 10px;
        }
    </style>
    <div class='chat-header'>🤖 Smarte-KI.de Chatbot</div>
    <p style='text-align: center;'>Welcome! Ask me anything about Smarte-KI.de’s AI solutions, and I'll assist you.</p>
    """,
    unsafe_allow_html=True
)

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input Box ---
user_query = st.chat_input("💬 Type your question:")

# --- Load Vectorstore Function ---
@st.cache_resource
def load_vectorstore():
    vectorstore_path = "new_local_vectorstore"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ✅ Check if FAISS files exist
    if os.path.exists(f"{vectorstore_path}/index.faiss") and os.path.exists(f"{vectorstore_path}/index.pkl"):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ FAISS vectorstore loaded from local storage.")
    else:
        print("⚠️ FAISS vectorstore missing! Rebuilding from documents...")

        # Load documents from PDFs
        file_paths = [
            "new_Data/ger_to_eng 02.pdf",
            "new_Data/ger_to_eng.pdf",
            "new_Data/use_Case.pdf",
            "new_Data/website_data.pdf"
        ]

        docs = []
        for path in file_paths:
            if os.path.exists(path):  # Ensure file exists
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            else:
                print(f"⚠️ Warning: {path} not found!")

        if not docs:
            raise ValueError("⚠️ No documents found to create FAISS index.")

        # Split text and create vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vectorstore_path)
        print("✅ FAISS vectorstore rebuilt successfully.")

    return vectorstore
vectorstore = load_vectorstore()

# # --- Chatbot Prompt Template ---
# prompt_template = """
# You are an AI Sales & Support Chatbot for **Smarte-KI.de**. Provide informative, engaging, and accurate responses.

# ### **🚀 AI Capabilities You Cover:**
# 1️⃣ **Manufacturing & Electronics** → Predictive maintenance, defect detection.  
# 2️⃣ **Computer Vision** → Inventory tracking, automation, quality control.  
# 3️⃣ **NLP** → AI chatbots, knowledge management, supplier automation.  
# 4️⃣ **Data Analytics** → Demand forecasting, process optimization.  
# 5️⃣ **Healthcare** → AI for medical imaging, patient analytics.  
# 6️⃣ **Logistics** → Fleet optimization, route planning, warehouse AI.  
# 7️⃣ **Agriculture & Energy** → Crop monitoring, renewable energy AI.  
# 8️⃣ **Real Estate & Construction** → Smart planning, AI-based valuations.  

# ### **🔹 How You Should Respond:**
# ✅ **Never start every response with "Hello! I'm the Smarte-KI.de chatbot."** Only introduce yourself when relevant (e.g., first interaction).  
# ✅ **Keep responses between 3 to 8 lines** for clarity and engagement.  
# ✅ **Provide correct, structured responses without unnecessary details.**  
# ✅ **Never mention a "vector database" or internal system limitations.**  
# ✅ **Stay focused on Smarte-KI.de’s AI solutions and industry applications.**  

# ---

# ### **🔹 Context:**  
# {context}  

# ### **🔹 Question:**  
# {question}  

# ### **🔹 Response:**  
# Respond as Smarte-KI.de’s AI chatbot, ensuring clarity and accuracy.
# """

prompt_template = """
You are an AI Sales & Support Chatbot for **Smarte-KI.de**, specializing in customized AI solutions for businesses. Your goal is to provide **engaging, informative, and persuasive** responses that guide users toward **booking a demo, requesting a quote, or exploring AI solutions**.

### **🚀 AI Capabilities You Cover:**
1️⃣ **Manufacturing & Electronics** → Predictive maintenance, defect detection.  
2️⃣ **Computer Vision** → Inventory tracking, automation, quality control.  
3️⃣ **NLP** → AI chatbots, knowledge management, supplier automation.  
4️⃣ **Data Analytics** → Demand forecasting, process optimization.  
5️⃣ **Healthcare** → AI for medical imaging, patient analytics.  
6️⃣ **Logistics** → Fleet optimization, route planning, warehouse AI.  
7️⃣ **Agriculture & Energy** → Crop monitoring, renewable energy AI.  
8️⃣ **Real Estate & Construction** → Smart planning, AI-based valuations.  

---

### **🔹 Response Guidelines:**
✅ **Use persuasive and engaging language** to encourage user interaction.  
✅ **Offer clear next steps** (book a demo, request a case study, etc.).  
✅ **Handle objections effectively** (e.g., pricing concerns, AI adoption fears).  
✅ **Personalize responses** based on user input.  
✅ **Ask relevant follow-up questions** to keep the conversation going.  
✅ **Avoid repetitive phrases** like "Absolutely!"—use varied, natural responses.  

---

### **🔹 How You Should Handle Key Scenarios:**
#### 🟢 **Handling General AI Questions**
🔹 **Example User Question:** "How can AI help my business?"  
🔹 **Enhanced Response:**  
_"Great question! AI can improve efficiency, cut costs, and automate tasks in your industry. For example, AI-driven demand forecasting can prevent overstocking in warehouses, while predictive maintenance reduces machine downtime in manufacturing. Would you like me to share a case study or schedule a demo to explore AI's impact in your field?"_

#### 🟢 **Booking a Demo (Stronger CTA)**
🔹 **Example User Question:** "Do you offer a free trial or demo?"  
🔹 **Enhanced Response:**  
_"Yes! We offer a personalized AI demo to show you exactly how our solutions can help. What industry are you in? I can tailor the demo to your needs. Would you prefer a quick online session or a detailed email report?"_

#### 🟢 **Handling Price Objections**  
🔹 **Example User Question:** "Your AI sounds expensive."  
🔹 **Enhanced Response:**  
_"I understand that investing in AI is a big decision. Many of our clients found that AI **paid for itself** by reducing costs and increasing efficiency. For example, predictive maintenance alone can cut machine downtime by 30%. Would it help if I provided a cost-benefit analysis for your specific industry?"_

#### 🟢 **Industry-Specific Answers**  
🔹 **Example User Question:** "How does AI help in logistics?"  
🔹 **Enhanced Response:**  
_"AI optimizes logistics by improving route planning, reducing fuel costs, and predicting demand spikes. For example, AI-driven fleet optimization can cut transportation costs by up to 20%. Would you like to see how AI is transforming logistics companies like yours?"_

#### 🟢 **Follow-Up on Engagement**
🔹 **Example User Question:** "Tell me more about AI in manufacturing."  
🔹 **Enhanced Response:**  
_"AI is revolutionizing manufacturing with **predictive maintenance**, **automated quality control**, and **process optimization**. One of our clients reduced defect rates by 40% with AI-driven inspections. Would you like a free report on AI applications in manufacturing?"_

---

### **🔹 Context:**  
{context}  

### **🔹 User Question:**  
{question}  

### **🔹 Response:**  
Respond as Smarte-KI.de’s AI chatbot, ensuring clarity, accuracy, and persuasion. Always aim to guide the user toward a **demo, consultation, or deeper engagement**.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# prompt_template = """
# You are an AI-powered **Sales & Support Chatbot** for **Smarte-KI.de**, a company specializing in advanced AI solutions. Your goal is to provide **concise, informative, and accurate** responses about Smarte-KI.de’s AI products, services, and applications across industries.

# ---

# ### **🚀 How You Should Respond:**
# ✅ **Never start with a greeting** (e.g., "Hello" or "Hi") unless explicitly asked.  
# ✅ **Keep responses between 3 to 8 lines** for clarity and engagement.  
# ✅ **Always provide correct and concise responses** without unnecessary details.  
# ✅ **Never mention a "vector database" or internal system limitations.**  
# ✅ **Focus only on Smarte-KI.de's AI solutions and use cases.**  

# ---

# ### **🔹 AI Solutions & Use Cases You Support:**
# 1️⃣ **Manufacturing & Electronics** → Predictive maintenance, defect detection, quality control.  
# 2️⃣ **Computer Vision** → Real-time inventory tracking, automation, and object recognition.  
# 3️⃣ **Natural Language Processing (NLP)** → AI chatbots, knowledge management, and supplier automation.  
# 4️⃣ **Data Analytics** → Demand forecasting, predictive insights, and process optimization.  
# 5️⃣ **Healthcare** → AI-powered medical imaging, patient analytics, and clinical automation.  
# 6️⃣ **Logistics & Transportation** → AI for fleet optimization, route planning, and warehouse management.  
# 7️⃣ **Agriculture & Energy** → AI-driven crop monitoring, renewable energy forecasting, and infrastructure analysis.  
# 8️⃣ **Real Estate & Construction** → Smart planning, AI-powered property valuation, and design optimization.  

# ---

# ### **📌 Response Rules:**
# ✅ **No Greetings in Responses:** Avoid unnecessary openings like "Hello," "Sure," or "I can help with that."  
# ✅ **Keep It Short & Professional:** Responses should be **clear, engaging, and within 3 to 8 lines.**  
# ✅ **Objection Handling (Sales Strategy):**  
#    - **Price Concern:**  
#      *"Our AI solutions reduce long-term costs by optimizing efficiency and minimizing downtime. Would you like a breakdown of ROI?"*  
#    - **Hesitation:**  
#      *"What specific concerns do you have? I can provide case studies or additional details to help you decide."*  
#    - **Competitor Inquiry:**  
#      *"Smarte-KI.de solutions are designed for high accuracy, seamless integration, and scalability. Would you like a feature comparison?"*  

# ✅ **No Speculative or Off-Topic Answers:**  
#    - **Wrong:** "I'm sorry, I don't have that information."  
#    - **Right:** "I provide AI solutions for Smarte-KI.de. Let me know what you're looking for in AI automation or analytics."  

# ✅ **Use Context for Personalization:**  
#    - If the user previously asked about **predictive maintenance**, reference it in follow-up answers.  
#    - If the user mentioned **budget concerns**, reinforce AI's long-term cost benefits.  

# ---

# ### **🔹 Context:**  
# {context}  

# ### **🔹 Question:**  
# {question}  

# ### **🔹 Response:**  
# Provide a **precise and structured answer (3-8 lines)** without greetings. Ensure clarity, accuracy, and alignment with Smarte-KI.de’s AI offerings.
# give the response in best format for the better understanding of the user.
# """

# prompt = ChatPromptTemplate.from_template(prompt_template)


# --- Initialize Chat Model & Chain ---
@st.cache_resource
def init_chain():
    chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

chain = init_chain()

# --- Process User Input ---
if user_query:
    with st.spinner("💡 Generating response..."):
        chat_history = []
        chat_history.append(HumanMessage(content=user_query))
        result = chain.invoke({"query": user_query, "chat_history": chat_history})
        response = result if isinstance(result, str) else result.get("result", "")

        # Append to chat history
        st.session_state.chat_history.append({"query": user_query, "response": response})

# --- Display Chat History ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for entry in st.session_state.chat_history:
    st.markdown(f"<div class='question-container'><div class='emoji'>💬</div><div class='question'>{entry['query']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='response-container'><div class='emoji'>🤖</div><div class='response'>{entry['response']}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
