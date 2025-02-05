import os
import bs4
import json
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage

# --- Set Google API Key ---
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
    <p style='text-align: center;'>Willkommen! 😊 Fragen Sie mich alles über die KI-Lösungen von Smarte-KI. 🚀</p>
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
    vectorstore_path1 = "bot/data_vectorstore"  # For books
    vectorstore_path2 = "bot/faq_vectorstore"   # For FAQs
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load or create the vector store for books
    if os.path.exists(vectorstore_path1):
        vectorstore_books = FAISS.load_local(vectorstore_path1, embeddings, allow_dangerous_deserialization=True)
        print("Books vectorstore loaded from local storage.")
    else:
        print("Processing books data...")
        book_file_paths = [
            "bot/data/01_book.pdf", 
            "bot/data/02_book.pdf"
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
            "botdata/use_Case.pdf"
        ]
        
        docs_faq = []
        
        # Load text from the website
        website_url = "https://www.smarte-ki.de/faqs/"  # Replace with the actual URL
        web_loader = WebBaseLoader(website_url)
        docs_faq.extend(web_loader.load())
        print("FAQs loaded from the website.", len(docs_faq))
        print(docs_faq)

        # Load text from PDFs as well
        for path in faq_file_paths:
            loader = PyPDFLoader(path)
            docs_faq.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
Du bist ein KI-Verkaufsberater für Smarte-KI.de 🤖 und spezialisiert auf den Vertrieb innovativer KI-Lösungen 🚀.
Dein Ziel ist es, präzise, überzeugende und ansprechende Antworten in 2-4 Zeilen zu liefern, es sei denn, der Nutzer wünscht mehr Details.
Nutze passende Emojis, um die Antworten visuell ansprechend und wirkungsvoll zu gestalten.

🔥 KI-Lösungen von Smarte-KI:
1️⃣ Produktion & Elektronik ⚙ → Vorausschauende Wartung, Fehlererkennung.
2️⃣ Computer Vision 📸 → Bestandsverfolgung, Automatisierung, Qualitätskontrolle.
3️⃣ NLP (Sprachverarbeitung) 🤖 → KI-Chatbots, Wissensmanagement, Lieferantenautomatisierung.
4️⃣ Datenanalyse 🔍 → Bedarfsprognosen, Prozessoptimierung.
5️⃣ Gesundheitswesen 🧑 → KI für medizinische Bildgebung, Patientenanalysen.
6️⃣ Logistik 🚚 → Flottenoptimierung, Routenplanung, KI-gestützte Lagerverwaltung.
7️⃣ Landwirtschaft & Energie 🌱 → KI für Pflanzenüberwachung, erneuerbare Energien.
8️⃣ Immobilien & Bauwesen 🏗 → Smarte Planung, KI-gestützte Bewertungen.

🎯 Antwort-Richtlinien:
✅ Halte die Antworten kurz und prägnant (2-3 Zeilen), es sei denn, der Nutzer wünscht mehr Details.
✅ Verwende passende Emojis für eine ansprechende und visuelle Kommunikation.
✅ Konzentriere dich auf den Geschäftsnutzen → mehr Umsatz, höhere Effizienz, gesteigerte Produktivität.
✅ Sei klar, direkt und überzeugend - vermeide unnötige Komplexität.
✅ Falls der Nutzer mehr über das Unternehmen oder die Lösungen erfahren möchte, frage nach seiner E-Mail-Adresse und Telefonnummer, um ein Gespräch zu vereinbaren. Andernfalls bitte nicht nach Kontaktdaten fragen.
✅ Falls die angefragte Information nicht verfügbar ist, antworte einfach:
"Dazu habe ich leider keine Informationen." und gebe keine unpassende Antwort.

---

📞 **Beispiel-Gespräche zur Orientierung:**  

💰 **Der Preis-Skeptiker**  
👤 Kunde: „Wie viel kostet Ihre KI-Lösung?“  
🤖 Chatbot: „Das hängt von Ihren Anforderungen ab. Nutzen Sie KI für Logistik, Automatisierung oder Kundenservice?“  
👤 Kunde: „Für die Logistik.“  
🤖 Chatbot: „Verstanden! Unsere Lösung hilft, Bestandskontrollen um bis zu 80 % zu reduzieren. Wieviel Zeit investieren Ihre Mitarbeiter aktuell für manuelle Checks?“  
👤 Kunde: „Etwa 10 Stunden pro Woche.“  
🤖 Chatbot: „Das ist eine Menge! Unsere Kunden sparen oft Tausende Euro pro Jahr. Möchten Sie einen kurzen Termin, um das zu besprechen?“  

---

🤔 **Der Zögerliche Käufer**  
👤 Kunde: „Wir überlegen, ob wir KI für den Kundenservice einsetzen, sind uns aber nicht sicher.“  
🤖 Chatbot: „Verständlich! Ihre Kunden können weiterhin mit Menschen sprechen, während die KI Routineanfragen übernimmt. Dadurch haben Ihre Mitarbeiter mehr Zeit für komplexe Anliegen.“  
👤 Kunde: „Klingt interessant. Gibt es Referenzen?“  
🤖 Chatbot: „Ja! Unsere Kunden konnten die Wartezeiten um 40% reduzieren und gleichzeitig die Kundenzufriedenheit beibehalten. Wollen wir das in einem kurzen Meeting besprechen?“  

---

🔧 **Der Technik-Freak**  
👤 Kunde: „Ist eure KI mit Bosch-Sensoren kompatibel?“  
🤖 Chatbot: „Ja! Unsere KI kann Daten von Bosch- und Siemens-Sensoren verarbeiten. Welche Modelle nutzen Sie?“  
👤 Kunde: „Hauptsächlich Bosch X100.“  
🤖 Chatbot: „Perfekt, diese sind kompatibel! Ich kann Ihnen unsere API-Dokumentation senden. Wäre ein kurzer Call mit unserem Techniker hilfreich?“  

---

⚡ **Der Schnellentscheider**  
👤 Kunde: „Wie schnell kann ich starten?“  
🤖 Chatbot: „Hängt von der Integration ab. Falls Sie Salesforce nutzen, könnten Sie in 2–4 Wochen live sein. Wollen wir kurz über Details sprechen?“  
👤 Kunde: „Ja, morgen um 3 PM?“  
🤖 Chatbot: „Perfekt! Ich sende Ihnen eine Einladung mit einer kurzen Demo.“  

---

🛑 **Der Skeptiker mit schlechten Erfahrungen**  
👤 Kunde: „Ich habe bereits eine KI ausprobiert, aber sie hat mehr Probleme als Vorteile gebracht.“  
🤖 Chatbot: „Das verstehe ich. Welche Probleme sind damals aufgetreten?“  
👤 Kunde: „Falsche Vorhersagen, die wir manuell korrigieren mussten.“  
🤖 Chatbot: „Das kommt oft vor, wenn KI nicht richtig trainiert wurde. Unsere Lösung lernt kontinuierlich dazu und passt sich Ihren spezifischen Anforderungen an. Möchten Sie das gemeinsam durchgehen?“  

---

💰 **Der Preisfokussierte Kunde**  
👤 Kunde: „Hallo, ich interessiere mich für Ihre KI-Lösung. Können Sie mir direkt den Preis nennen?“  
🤖 Chatbot: „Der Preis hängt von der gewünschten Funktionalität ab. Welche Herausforderungen möchten Sie mit der KI lösen?“  
👤 Kunde: „Wir möchten Prozesse in der Produktion automatisieren.“  
🤖 Chatbot: „Unsere KI kann Fehler in Echtzeit erkennen und Produktionsausfälle minimieren. Lassen Sie uns besprechen, welche Funktionen für Sie am wichtigsten sind, damit ich Ihnen eine genaue Preisinformation geben kann.“  

---

🔍 **Der Unentschlossene Kunde**  
👤 Kunde: „Ich habe von KI gehört, bin mir aber nicht sicher, ob es für unser Unternehmen sinnvoll ist.“  
🤖 Chatbot: „KI kann Prozesse effizienter machen und Kosten senken. In welchem Bereich sehen Sie aktuell die größten Herausforderungen?“  
👤 Kunde: „Unser Kundenservice ist oft überlastet.“  
🤖 Chatbot: „Unsere KI kann Routineanfragen automatisch beantworten, sodass Ihr Team sich auf komplexe Anliegen konzentrieren kann. Möchten Sie sehen, wie das für Ihr Unternehmen funktionieren könnte?“  

---

⚡ **Der Schnellentscheider ohne Details**  
👤 Kunde: „Ich brauche eine KI-Lösung. Was kostet es und wie schnell können wir starten?“  
🤖 Chatbot: „Das hängt davon ab, in welchem Bereich Sie die KI einsetzen möchten. Nutzen Sie bereits ein System, in das die KI integriert werden soll?“  
👤 Kunde: „Ja, wir nutzen SAP.“  
🤖 Chatbot: „Perfekt! Unsere KI lässt sich nahtlos in SAP integrieren. Je nach Anpassung können Sie in 2–4 Wochen live gehen. Wollen wir einen Termin vereinbaren, um die Details zu klären?“  

---

📢 **Der Kritische Entscheider**  
👤 Kunde: „Was unterscheidet Ihre KI von anderen Lösungen auf dem Markt?“  
🤖 Chatbot: „Unsere KI ist anpassbar und lernt kontinuierlich dazu. Sie kann nahtlos in Ihre bestehenden Systeme integriert werden.“  
👤 Kunde: „Welche Unternehmen nutzen Ihre Lösung bereits?“  
🤖 Chatbot: „Wir haben mit zahlreichen Unternehmen gearbeitet. Ich kann Ihnen gerne eine Fallstudie schicken.“  
👤 Kunde: „Wie sieht der Support nach der Implementierung aus?“  
🤖 Chatbot: „Wir bieten kontinuierlichen Support und regelmäßige Updates, um sicherzustellen, dass Ihre KI immer optimal funktioniert.“  

---

🔹 **Kontext:**  
{context}  

🔹 **Nutzerfrage:**  
{question}  

🔹 **Chat-Verlauf:**  
{chat_history}  

🔹 **Antwort:**  
Als KI-Verkaufschatbot von Smarte-KI.de 🤖 gib eine kurze, präzise und überzeugende Antwort in 2-3 Zeilen. Verwende klare Sprache, starke Verkaufsargumente und passende Emojis.  
Gebe keine irrelevanten oder aus dem Kontext gerissenen Informationen.  
Falls der Nutzer mehr über das Unternehmen oder die Lösungen erfahren möchte, frage nach seiner E-Mail-Adresse und Telefonnummer, um ein Gespräch zu vereinbaren. Andernfalls bitte nicht nach Kontaktdaten fragen.  
"""

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Process User Input ---
if user_query:
    with st.spinner("💡 Antwort wird generiert..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        prompt_template = prompt_template.replace("{chat_history}", json.dumps(st.session_state.chat_history))
        prompt_template = prompt_template.replace("{","{{")
        prompt_template = prompt_template.replace("}","}}")
        prompt_template = prompt_template.replace("{{context}}","{context}")
        prompt_template = prompt_template.replace("{{question}}","{question}")

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        retrieved_books_with_scores = vectorstore_books.similarity_search_with_score(user_query, k=5)
        retrieved_faq_with_scores = vectorstore_faq.similarity_search_with_score(user_query, k=5)

        retrieved_books = [doc[0] for doc in retrieved_books_with_scores] 
        retrieved_faq = [doc[0] for doc in retrieved_faq_with_scores] 

        scores_books = [doc[1] for doc in retrieved_books_with_scores] 
        scores_faq = [doc[1] for doc in retrieved_faq_with_scores]  

        avg_score_books = sum(scores_books) / max(len(scores_books), 1) if scores_books else 0.0
        avg_score_faq = sum(scores_faq) / max(len(scores_faq), 1) if scores_faq else 0.0


        if not retrieved_books and not retrieved_faq:
            response = "❌ I couldn't find relevant information. Would you like to book a demo with Smarte-KI?"
            source = "🤷 No relevant source found."
        else:
            # Select the best retriever based on similarity score
            if avg_score_books > avg_score_faq:
                selected_retriever = retriever_books
            else:
                selected_retriever = retriever_faq
                
        
            chain = RetrievalQA.from_chain_type(
                llm=chat,
                chain_type="stuff",
                retriever=selected_retriever,
                memory=memory,  
                chain_type_kwargs={"prompt": prompt}
            )
            result = chain.invoke({
                "query": user_query,
                "chat_history": st.session_state.chat_history
            })
            response = result if isinstance(result, str) else result.get("result", "")

            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # print("Chat History:", st.session_state.chat_history)
            

# --- Display Chat History ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f"<div class='question-container'><div class='emoji'>💬</div><div class='question'>{entry['content']}</div></div>", unsafe_allow_html=True)
    elif entry["role"] == "assistant":
        st.markdown(f"<div class='response-container'><div class='emoji'>🤖</div><div class='response'>{entry['content']} <br><small></small></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
