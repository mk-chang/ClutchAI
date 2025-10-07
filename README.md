# ClutchAI 🧠🏈  
**AI-powered fantasy sports assistant built with Retrieval-Augmented Generation (RAG)**  

ClutchAI connects to the **Yahoo Fantasy Sports API** to give you real-time, context-aware answers about your fantasy league — powered by LLMs and vector-based retrieval.  

## 🚀 Features
- **Yahoo Fantasy Integration:** Securely connect your league, team, and player data via OAuth.  
- **Natural Language Q&A:** Ask questions like *“Who should I start at FLEX this week?”* or *“How did my matchup go?”*  
- **Contextual Intelligence:** Uses a RAG system to ground responses in your actual league data.  
- **Structured + Semantic Retrieval:** Combines factual data (scores, rosters) with contextual summaries (player notes, matchups).  
- **Explainable Insights:** Every answer includes source context from your league.  

## 🧩 Tech Stack
- **Backend:** Python / FastAPI  
- **Data Source:** Yahoo Fantasy Sports API  
- **LLM Layer:** OpenAI or local LLM (configurable)  
- **Vector Store:** Pinecone / Qdrant / Weaviate  
- **Auth:** Yahoo 3-legged OAuth2  
- **Database:** Google Cloud PostgresSQL 

## ⚙️ Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/clutchai.git
   cd clutchai
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables
   ```makefile
    YAHOO_CLIENT_ID=
    YAHOO_CLIENT_SECRET=
    OPENAI_API_KEY=
    VECTOR_DB_URL=
   ```
4. Run the app
   ```bash
   uvicorn app.main:app --reload
   ```

## Authorize your Yahoo account
Visit /auth/yahoo to connect your fantasy league.

## 💬 Example Queries
- "Show my team’s performance from last week."
- "Who are the top waiver pickups this week?"
- "Compare my RBs to the league average."
- "Should I start Joe Mixon or Austin Ekeler?"

## 🧠 How It Works
Data Retrieval: Pulls league + team data via Yahoo API.
Embedding: Converts textual summaries into vector embeddings.
Storage: Saves structured data in Postgres and embeddings in a vector DB.
RAG Pipeline: On query, retrieves relevant context and augments the LLM prompt.
Response Generation: LLM generates grounded, explainable insights.

## 🔒 Security
OAuth tokens are encrypted and stored securely.
Private league data is never shared outside your account
 
