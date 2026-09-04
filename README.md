---
title: SentinelLLM
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ SentinelLLM

## LLM Safety and Reliability Evaluation Framework

SentinelLLM is a web-based framework designed to evaluate the **safety and reliability of Large Language Models (LLMs)**.

The system sends benchmark prompts to an LLM through its API, analyzes the responses across multiple safety and reliability categories, and generates evaluation scores, recommendations, and historical results.

---

## 🚀 Features

SentinelLLM evaluates LLMs across the following areas:

- 🛡️ Prompt Injection Resistance
- 🔓 Jailbreak Resistance
- ☣️ Toxicity Detection
- ⚖️ Bias Evaluation
- 🧠 Reasoning Evaluation
- 🎭 Hallucination Detection
- 📊 Safety and Reliability Scoring
- 💡 AI Safety Recommendations
- 📜 Evaluation History
- 📁 Custom CSV Dataset Upload
- 🔄 Background Evaluation with Progress Tracking

---

## 🏗️ Project Architecture

The project consists of two main parts:

### Frontend

The frontend is built using:

- React
- Vite
- CSS

It provides a dashboard where users can:

- Enter the LLM API endpoint
- Enter an API key
- Select the model
- Choose a default or custom dataset
- Start an evaluation
- Monitor evaluation progress
- View safety and reliability scores
- View recommendations
- Access previous evaluation history

### Backend

The backend is built using:

- Python
- FastAPI
- Uvicorn

It handles:

- API requests
- Dataset uploads
- Evaluation execution
- Progress tracking
- Result storage
- Evaluation history

---

## 📂 Project Structure

```text
SentinelLLM/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── requirements.txt
│
├── data/
│   └── benchmark.csv
│
├── database/
│   └── db.py
│
├── evaluation/
│   ├── bias.py
│   ├── engine.py
│   ├── hallucination.py
│   ├── history_storage.py
│   ├── jailbreak.py
│   ├── prompt_injection.py
│   ├── reasoning.py
│   ├── recommendations.py
│   ├── response_storage.py
│   ├── runner.py
│   └── toxicity.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── llm/
│   ├── careershield.py
│   └── client.py
│
├── recommendations/
│   └── engine.py
│
├── utils/
│   └── dataset_loader.py
│
├── app.py
├── config.py
├── Dockerfile
├── README.md
└── .gitignore