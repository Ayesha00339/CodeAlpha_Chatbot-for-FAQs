# 🎓 University FAQ Chatbot

A smart, AI-powered chatbot designed to help **university students** get instant answers to their everyday questions — without waiting in queues or searching through long documents. Whether you need information about **admissions, fee structure, scholarships, exams, hostel, or campus facilities**, this chatbot understands your question in natural language and returns the most relevant answer within seconds.

Built as an **AI Internship Task** for **CodeAlpha**, this project demonstrates the practical use of **NLP techniques** — including tokenization, lemmatization, TF-IDF vectorization, and cosine similarity — to build a real-world conversational assistant.

---

## 📋 Features

- ✅ **30 FAQs** covering Admissions, Fees, Scholarships, Academics, Exams, Hostel, Campus Facilities, and Student Services
- ✅ **NLP Preprocessing** — Tokenization, Stopword Removal, Lemmatization using NLTK
- ✅ **TF-IDF Vectorization** with **Bigram (1–2) support** for better contextual matching
- ✅ **Cosine Similarity** — Finds the most relevant FAQ for any user query
- ✅ **Intent Detection** — Handles greetings, thanks, and farewells separately with randomized responses
- ✅ **Match Confidence Score** — Displays how closely the answer matches the query (e.g. Match: 59%)
- ✅ **Matched Question Display** — Shows the exact FAQ question that was matched (📎 Matched: "...")
- ✅ **Sample Questions Sidebar** — 6 clickable quick-fill questions for instant demo
- ✅ **Native Streamlit Chat UI** — Clean bubble-style conversation using `st.chat_message()`
- ✅ **Clear Chat** button to reset the full conversation
- ✅ **Session Stats** — Tracks questions asked, total FAQs in database, and NLP method used
- ✅ **Topic Chips** — Visual sidebar tags showing all covered topics
- ✅ **Dark Mode UI** — Custom styled dark theme with green accent colors
- ✅ **Low Confidence Fallback** — Friendly suggestion message when no good match is found (threshold: 0.15)

---

## 🛠️ Technologies Used

| Library         | Purpose                                            |
|-----------------|--------------------------------------------------  |
| `streamlit`     | Web UI / Chat Interface                            |
| `nltk`          | Tokenization, Stopword Removal, Lemmatization      |
| `scikit-learn`  | TF-IDF Vectorizer + Cosine Similarity              |
| `numpy`         | Array operations for similarity score extraction   |
| `random`        | Randomized responses for greetings/thanks/farewell |
| `string`        | Punctuation removal during preprocessing           |

---

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the app
```bash
streamlit run app.py
```

### Step 3: Open in browser
```
http://localhost:8501
```

---

## 🧠 How It Works

```
User Input
    │
    ▼
Intent Detection
  • Greeting  → Random greeting response
  • Thanks    → Random thanks response
  • Farewell  → Random farewell response
  • FAQ       → Continue to NLP pipeline
    │
    ▼
NLP Preprocessing
  • Lowercase conversion
  • Tokenize  (NLTK word_tokenize)
  • Remove punctuation & stopwords
  • Lemmatize (WordNetLemmatizer)
    │
    ▼
TF-IDF Vectorization (bigrams: 1–2)
    │
    ▼
Cosine Similarity against all 30 FAQ questions
    │
    ▼
Best Match Selection
  • Score ≥ 0.15 → Return answer + matched question + confidence %
  • Score < 0.15 → Return friendly fallback message
    │
    ▼
Display in Chat UI (st.chat_message)
```

---

## 📁 Project Structure

```
faq_chatbot/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📚 FAQ Categories

| Category              | Count |
|-----------------------|-------|
| Admissions            | 5     |
| Fees & Scholarships   | 4     |
| Academics             | 6     |
| Campus & Facilities   | 5     |
| Exams & Results       | 4     |
| Student Services      | 5     |
| **Total**             | **29**|

---

## 💬 Supported Intents (Beyond FAQs)

| Intent    | Trigger Words                                                      |
|-----------|--------------------------------------------------------------------|
| Greeting  | hello, hi, hey, assalam, salam, good morning, good evening         |
| Thanks    | thanks, thank you, shukriya, shukran, jazakallah, helpful          |
| Farewell  | bye, goodbye, khuda hafiz, allah hafiz, see you, exit, quit        |

---

## 👩‍💻 Author

**AYESHA BUKHARI**
*AI Internship Task — CodeAlpha*
*FAQ Chatbot using NLP & Cosine Similarity*
