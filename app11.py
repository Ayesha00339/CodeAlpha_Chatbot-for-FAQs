import streamlit as st
import nltk
import string
import time
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─── NLTK Setup ────────────────────────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

# ─── FAQ Dataset (University / Academic Topic) ─────────────────────────────────
FAQ_DATA = [
    # Admission
    ("How can I apply for admission?",
     "You can apply for admission by visiting our official university website, filling out the online application form, and submitting required documents such as your transcripts, CNIC/B-Form, and passport-size photo."),
    ("What is the last date for admission?",
     "Admissions typically open in August for Fall semester and February for Spring semester. The last date is usually announced on the official university website. Check regularly to avoid missing it!"),
    ("What documents are required for admission?",
     "Required documents usually include: Matric & Intermediate certificates, CNIC or B-Form copy, passport-size photos, migration certificate (if applicable), and equivalence certificate for foreign degrees."),
    ("Is there an entry test for admission?",
     "Yes, most programs require you to pass a university entry test or NTS/GAT score. Some programs may also accept SAT scores. Check your specific department's requirements on the university website."),
    ("Can I apply for multiple programs simultaneously?",
     "Yes, you can apply for multiple programs but you will need to pay the application fee for each program separately. Selection is based on merit and test performance."),

    # Fee & Scholarship
    ("What is the fee structure?",
     "Fee varies by program and department. Generally, BS programs range from PKR 30,000–60,000 per semester. You can find the complete fee structure on the university's official fee page or by contacting the accounts department."),
    ("Are scholarships available?",
     "Yes! The university offers need-based and merit-based scholarships. HEC also offers scholarships such as the Need-Based Scholarship (NBS) program. Apply through the financial aid office."),
    ("How can I get a fee concession?",
     "Fee concession is available for students with financial hardship. You need to submit an application to the Student Affairs office along with supporting documents like income certificate and utility bills."),
    ("When is the fee submission deadline?",
     "Fee is usually due within the first 2 weeks of each semester. Late payment may result in a fine or registration cancellation. Always check the academic calendar for exact dates."),

    # Academics
    ("What is the grading system?",
     "The university follows a standard GPA system: A (4.0), A- (3.7), B+ (3.3), B (3.0), B- (2.7), C+ (2.3), C (2.0), D (1.0), F (0.0). A minimum CGPA of 2.0 is required to stay enrolled."),
    ("How many credit hours are required to complete a BS degree?",
     "A BS (4-year) degree typically requires 130–136 credit hours depending on your department. This includes core courses, electives, and the Final Year Project (FYP)."),
    ("What is the attendance policy?",
     "Students must maintain at least 75% attendance in each course. Falling below this threshold may result in being debarred from the final examination. Medical leaves may be considered with proper documentation."),
    ("How do I register for courses?",
     "Course registration is done online through the Student Portal (LMS) at the start of each semester. Your academic advisor must approve your course selection before the deadline."),
    ("Can I repeat a course to improve my grade?",
     "Yes, you can repeat a course if you scored a D or F. The new grade will replace the old one in your transcript. However, there is a limit on how many courses you can repeat."),
    ("What is the FYP?",
     "FYP stands for Final Year Project. It is a capstone research/development project completed in the final year of your degree, usually spanning two semesters. It requires a supervisor and formal evaluation."),

    # Campus & Facilities
    ("Is hostel facility available?",
     "Yes, the university provides separate hostel facilities for male and female students. Hostel allocation is done on a first-come, first-served basis. Contact the hostel office for availability and fees."),
    ("Is there a transport service?",
     "Yes, the university offers a transport service covering major routes. Route cards are issued at the start of the semester. Contact the transport office for route details and monthly charges."),
    ("What library resources are available?",
     "The university library provides access to thousands of books, research journals, IEEE Xplore, Springer, and other digital databases. Students can borrow books using their student ID card."),
    ("Is Wi-Fi available on campus?",
     "Yes, Wi-Fi is available throughout the campus including classrooms, labs, library, and cafeteria. Students can connect using their university-issued credentials."),
    ("Are there sports facilities?",
     "Yes! The campus has cricket ground, badminton courts, table tennis, gymnasium, and indoor games. Annual sports events are also held. Contact the Student Affairs department to join any sports team."),

    # Exams & Results
    ("When are final exams held?",
     "Final exams are held at the end of each semester — usually in January (Fall) and June (Spring). The exact date sheet is announced 2–3 weeks before exams on the LMS and notice boards."),
    ("How do I check my result?",
     "Results are published on the official Student Portal/LMS after each exam. You will receive notifications via your university email. Contact the Examination Department for any discrepancies."),
    ("What happens if I fail a subject?",
     "If you fail a subject (grade F), you must retake it in the next available semester. Repeated failures in core courses may lead to academic probation. Seek guidance from your academic advisor."),
    ("Can I apply for rechecking of my paper?",
     "Yes, you can apply for paper rechecking within 7 days of result announcement. Submit an application to the Examination Department along with the required fee. Results of rechecking are final."),

    # Student Services
    ("How do I get my transcript?",
     "You can request your official transcript from the Examination Department by submitting an application and paying the required fee. Processing usually takes 3–5 working days."),
    ("How do I get a student ID card?",
     "Student ID cards are issued during the first week of enrollment. Visit the Registration Office with your admission letter and a passport-size photo. Replacement cards have a small fee."),
    ("Is there a career counseling service?",
     "Yes, the Career Development Center (CDC) offers career counseling, CV workshops, mock interviews, and job placement assistance. Regular industry visits and career fairs are also organized."),
    ("How can I join university societies?",
     "The university has various academic and social societies (CS Society, Literary Club, Debating Society, etc.). Joining is open to all students — check the Student Affairs notice board or LMS for announcements."),
    ("What should I do if I have a complaint?",
     "For academic complaints, contact your Head of Department. For administrative issues, visit the Student Affairs Office. You can also submit formal complaints through the university's online complaint portal."),
]

GREETINGS = ["hello", "hi", "hey", "assalam", "salam", "assalamualaikum", "good morning", "good evening", "good afternoon"]
GREETING_RESPONSES = [
    "Wa Alaikum Assalam! 😊 Welcome to the University FAQ Chatbot! How can I assist you today?",
    "Hello! 👋 I'm your University FAQ assistant. Ask me anything about admissions, fees, exams, or campus life!",
    "Hi there! 😊 I'm here to help with all your university-related questions. What would you like to know?",
]

THANKS = ["thanks", "thank you", "shukriya", "shukran", "jazakallah", "great", "helpful"]
THANKS_RESPONSES = [
    "You're welcome! 😊 Feel free to ask if you have more questions.",
    "Happy to help! JazakAllah Khair 🌟",
    "Anytime! If you have more queries, I'm always here. 😊",
]

FAREWELLS = ["bye", "goodbye", "khuda hafiz", "allah hafiz", "see you", "exit", "quit"]
FAREWELL_RESPONSES = [
    "Allah Hafiz! 🌟 Feel free to return if you have more questions.",
    "Goodbye! Best of luck with your studies! 📚",
    "Take care! May Allah bless you with success! 🤲",
]

# ─── NLP Preprocessing ────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Preprocess all FAQ questions
faq_questions = [q for q, _ in FAQ_DATA]
faq_answers = [a for _, a in FAQ_DATA]
preprocessed_questions = [preprocess(q) for q in faq_questions]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

def get_best_match(user_query: str, threshold: float = 0.15):
    clean_query = preprocess(user_query)
    query_vec = vectorizer.transform([clean_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    return best_idx, best_score

def classify_intent(text: str) -> str:
    lower = text.lower().strip()
    if any(g in lower for g in GREETINGS):
        return "greeting"
    if any(t in lower for t in THANKS):
        return "thanks"
    if any(f in lower for f in FAREWELLS):
        return "farewell"
    return "faq"

def get_response(user_input: str):
    intent = classify_intent(user_input)
    if intent == "greeting":
        return random.choice(GREETING_RESPONSES), None, None
    if intent == "thanks":
        return random.choice(THANKS_RESPONSES), None, None
    if intent == "farewell":
        return random.choice(FAREWELL_RESPONSES), None, None

    idx, score = get_best_match(user_input)
    if score < 0.15:
        return (
            "🤔 I'm not sure about that. Could you rephrase your question? "
            "You can ask about admissions, fees, exams, scholarships, hostel, or campus facilities.",
            None, score
        )
    return faq_answers[idx], faq_questions[idx], score


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="University FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

/* Base */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f1724 0%, #111d2e 50%, #0a1628 100%);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f35 0%, #091827 100%) !important;
    border-right: 1px solid rgba(16,185,129,0.15);
}
[data-testid="stHeader"] { background: transparent; }

/* Title */
.main-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #10b981, #34d399, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    letter-spacing: 0.05em;
}

/* Chat bubbles */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 0.5rem 0;
}
.msg-row-user { display: flex; justify-content: flex-end; }
.msg-row-bot  { display: flex; justify-content: flex-start; align-items: flex-start; gap: 0.6rem; }

.bubble-user {
    background: linear-gradient(135deg, #10b981, #059669);
    color: #fff;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 2px 12px rgba(16,185,129,0.3);
}
.bubble-bot {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0;
    padding: 0.85rem 1.1rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
    backdrop-filter: blur(8px);
}
.bot-avatar {
    width: 34px; height: 34px; min-width: 34px;
    background: linear-gradient(135deg, #10b981, #059669);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.matched-q {
    font-size: 0.75rem;
    color: #10b981;
    margin-top: 0.4rem;
    opacity: 0.8;
}
.score-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981;
    font-size: 0.7rem;
    padding: 1px 7px;
    border-radius: 20px;
    margin-left: 6px;
}

/* Input */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.2) !important;
}

/* Sidebar */
.sidebar-section {
    background: rgba(16,185,129,0.05);
    border: 1px solid rgba(16,185,129,0.15);
    border-radius: 10px;
    padding: 0.9rem;
    margin-bottom: 0.8rem;
}
.sidebar-title {
    color: #10b981;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.topic-chip {
    display: inline-block;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.2);
    color: #6ee7b7;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    color: #94a3b8;
    font-size: 0.82rem;
    padding: 3px 0;
}
.stat-val { color: #10b981; font-weight: 600; }

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Scrollable chat area */
.chat-scroll {
    max-height: 520px;
    overflow-y: auto;
    padding-right: 4px;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-track { background: transparent; }
.chat-scroll::-webkit-scrollbar-thumb { background: rgba(16,185,129,0.3); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "bot",
        "content": "Assalam-o-Alaikum! 🎓 Welcome to the **University FAQ Chatbot**!\n\nI can help you with questions about **admissions, fees, scholarships, exams, hostel, campus facilities,** and much more. What would you like to know?",
        "matched_q": None,
        "score": None
    })
if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.4rem;">🎓 FAQ Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">University Help Desk</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">📚 Topics I Cover</div>
        <span class="topic-chip">🎯 Admissions</span>
        <span class="topic-chip">💰 Fee & Scholarships</span>
        <span class="topic-chip">📝 Academics</span>
        <span class="topic-chip">🏠 Hostel</span>
        <span class="topic-chip">📋 Exams</span>
        <span class="topic-chip">🚌 Transport</span>
        <span class="topic-chip">📚 Library</span>
        <span class="topic-chip">🏆 Sports</span>
        <span class="topic-chip">🎓 Transcript</span>
        <span class="topic-chip">💼 Career</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">📊 Session Stats</div>
        <div class="stat-row"><span>Questions Asked</span><span class="stat-val">{st.session_state.question_count}</span></div>
        <div class="stat-row"><span>FAQs in Database</span><span class="stat-val">{len(FAQ_DATA)}</span></div>
        <div class="stat-row"><span>NLP Method</span><span class="stat-val">TF-IDF + Cosine</span></div>
        <div class="stat-row"><span>Preprocessing</span><span class="stat-val">NLTK</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">💡 Sample Questions</div>
    </div>
    """, unsafe_allow_html=True)

    sample_qs = [
        "How to apply for admission?",
        "What documents are needed?",
        "Are scholarships available?",
        "What is the grading system?",
        "Is hostel available?",
        "How to check my result?",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_sample = q
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [{
            "role": "bot",
            "content": "Chat cleared! 😊 How can I help you with university-related questions?",
            "matched_q": None,
            "score": None
        }]
        st.session_state.question_count = 0
        st.rerun()

    st.markdown("""
    <div style="text-align:center; color:#334155; font-size:0.75rem; margin-top:1rem;">
        Powered by NLTK · TF-IDF · Cosine Similarity<br>
        <span style="color:#10b981;">AI Internship Task — CodeAlpha</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 University FAQ Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask anything about university life · Powered by NLP</div>', unsafe_allow_html=True)

# Handle sample question click
if "pending_sample" in st.session_state:
    user_input = st.session_state.pending_sample
    del st.session_state.pending_sample

    st.session_state.messages.append({"role": "user", "content": user_input, "matched_q": None, "score": None})
    answer, matched_q, score = get_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": answer, "matched_q": matched_q, "score": score})
    st.session_state.question_count += 1
    st.rerun()

# Render chat messages using Streamlit's native chat_message API
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🎓"):
            content = msg["content"].replace("**", "")
            st.markdown(content)
            if msg.get("matched_q") and msg.get("score") is not None:
                pct = int(msg["score"] * 100)
                st.markdown(
                    f'<div class="matched-q">📎 Matched: "{msg["matched_q"]}" '
                    f'<span class="score-badge">Match: {pct}%</span></div>',
                    unsafe_allow_html=True,
                )

# Chat input
user_input = st.chat_input("Ask your university question here... (e.g. How to apply for admission?)")
if user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input, "matched_q": None, "score": None})
    answer, matched_q, score = get_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": answer, "matched_q": matched_q, "score": score})
    st.session_state.question_count += 1
    st.rerun()
