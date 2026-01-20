# FloatChatAI - AI-Powered Ocean Data Explorer

## 📖 About the Project

FloatChatAI is an intelligent oceanographic data analysis platform that allows you to query and analyze real ARGO float data using natural language. Instead of writing complex database queries or processing raw scientific data files manually, simply ask questions like *"Show me temperature profiles for float 1900121"* and get instant, AI-powered responses with interactive visualizations.

**ARGO floats** are autonomous instruments that drift with ocean currents, measuring temperature, salinity, pressure, and other oceanographic parameters at various depths. This project makes this valuable scientific data accessible to everyone.

## 🔧 How It Works

FloatChatAI combines multiple cutting-edge technologies to deliver intelligent ocean data insights:

1. **Natural Language Processing**: Your questions are processed by Google's Gemini AI, which understands scientific queries and oceanographic terminology.

2. **RAG Pipeline** (Retrieval-Augmented Generation): The system retrieves relevant oceanographic documentation and scientific context from a vector database (ChromaDB) to provide accurate, domain-specific answers.

3. **MCP Server** (Model Context Protocol): A specialized server provides 20+ oceanographic tools (get trajectories, analyze profiles, find floats in regions, etc.) that the AI can intelligently invoke.

4. **Database Integration**: Real ARGO float data is stored in PostgreSQL and queried efficiently based on your natural language requests.

5. **Visualization Engine**: Responses include interactive charts (Plotly) and maps (Folium) showing depth profiles, trajectories, and spatial distributions.

6. **Web Interface**: A user-friendly Streamlit dashboard and FastAPI backend provide seamless interaction.

**Example Flow**: You ask *"What's the average temperature at 500m depth near Hawaii?"* → AI understands intent → Retrieves scientific context → Queries database for Hawaiian region profiles → Calculates statistics at 500m → Generates explanatory text and visualization → Returns complete answer.

---

## 🚀 Installation & Setup

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or higher** ([Download Python](https://www.python.org/downloads/))
- **PostgreSQL 13+** installed and running on port 5433 ([Download PostgreSQL](https://www.postgresql.org/download/))
- **Google Gemini API Key** ([Get API Key](https://makersuite.google.com/app/apikey))
- **4GB+ RAM** recommended
- **Windows/Linux/Mac** operating system

### Step 1: Clone or Download the Project

```bash
cd FloatChatAI
```

### Step 2: Create Virtual Environment

Open your terminal/command prompt in the project directory:

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages

Install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (API server)
- Streamlit (Web dashboard)
- PostgreSQL drivers (asyncpg, psycopg2)
- Google Gemini AI (google-generativeai)
- LangChain (AI framework)
- ChromaDB (Vector database)
- Plotly & Folium (Visualizations)
- pandas, numpy, xarray (Data processing)
- NetCDF4 (Scientific data files)

**Installation time**: 5-10 minutes depending on your internet speed.

### Step 4: Configure Environment Variables

**Windows:**
```powershell
copy .env.example .env
```

**Linux/Mac:**
```bash
cp .env.example .env
```

Now edit the `.env` file with your actual credentials:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5433
DB_USER=postgres
DB_PASSWORD=your-actual-database-password
DB_NAME=argo_ocean_data

# Google Gemini API Keys
GOOGLE_API_KEY=your-google-api-key-from-ai-studio
GOOGLE_API_KEY_PRIMARY=your-primary-api-key
GOOGLE_API_KEY_FALLBACK=your-fallback-api-key
```

**How to get Google Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in your `.env` file

### Step 5: Setup PostgreSQL Database

Ensure PostgreSQL is running, then initialize the database:

**Windows:**
```powershell
# Check if PostgreSQL is running
Get-Service postgresql*

# If not running, start it from Services or run:
# net start postgresql-x64-16  (adjust version number)

# Setup database schema and tables
python database/setup.py
```

**Linux/Mac:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# If not running, start it
sudo systemctl start postgresql

# Setup database schema and tables
python database/setup.py
```

The setup script will:
- Create the `argo_ocean_data` database
- Create all necessary tables (floats, profiles, measurements)
- Set up indices for optimal query performance
- Verify the connection

### Step 6: Run the Application

You can run the application in two ways:

#### **Option A: Streamlit Dashboard (Recommended for beginners)**

```bash
streamlit run frontend/enhanced_streamlit_app.py
```

This will:
- Start the web interface at `http://localhost:8501`
- Open automatically in your default browser
- Provide a chat-like interface for queries
- Display interactive visualizations

#### **Option B: FastAPI Backend (For developers)**

```bash
python api/main.py
```

This will:
- Start the API server at `http://localhost:8000`
- Provide REST endpoints for programmatic access
- Auto-generate API documentation at `http://localhost:8000/docs`
- Enable integration with other applications

---

## ✅ Verification

To verify everything is working correctly:

1. **Check Database Connection:**
   ```bash
   python database/config_tester.py
   ```

2. **Run Security Check:**
   ```bash
   python testing/security_check.py
   ```

3. **Test MCP Setup:**
   ```bash
   python testing/verify_mcp_setup.py
   ```

---

## 🎯 Quick Start Guide

Once the application is running, try these example queries:

**In Streamlit Dashboard:**
- *"Show me temperature profiles for float 1900121"*
- *"What floats are active in the Pacific Ocean?"*
- *"Plot salinity vs depth for the last month"*
- *"Find all floats near 45°N, 120°W"*

**Using FastAPI (with curl):**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show temperature data", "limit": 50}'
```

---

