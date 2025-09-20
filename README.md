This project implements a **multi-agent system** powered by **RAG (Retrieval-Augmented Generation)** for collaborative reasoning and evaluation.

## 📦 Project Structure

├── rag_system_build.py      # Build and initialize the RAG knowledge base

├── agent_*.py               # Individual agent scripts (API and URL need to be configured)

├── evaluation.py            # Evaluation script for the final results

└── README.md

## 🚀 Usage

### 1️⃣ Build the RAG System
Run the following script to construct the RAG knowledge base and initialize the retriever:
\`\`\`bash
python rag_system_build.py
\`\`\`

### 2️⃣ Configure Agents
Open each \`agent_xxx.py\` file and fill in your own API key, base URL, or any other required parameters:
\`\`\`python
API_KEY = "your_api_key_here"
BASE_URL = "https://your-model-url.com"
\`\`\`

### 3️⃣ Run the Agents
Execute the agents one by one to complete their reasoning tasks:

### 4️⃣ Evaluate Results
Finally, run the evaluation script to assess the outputs from all agents:
\`\`\`bash
python evaluation.py
\`\`\`
