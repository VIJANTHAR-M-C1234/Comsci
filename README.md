---
title: NCERT AI Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# NCERT AI Assistant

An educational AI system that helps school students (Class 6–12) solve their NCERT textbook doubts.
The chatbot uses Retrieval-Augmented Generation (RAG) powered by **Qwen2.5-7B-Instruct** via Hugging Face Inference API.
It supports multiple Indian languages like Tamil, Hindi, Telugu, Kannada, and more.

👉 **[Live App](https://huggingface.co/spaces/Parasuramane24/NCERT-AI-Assistant)**

## Features
- 🔍 Context-aware RAG retrieval from NCERT textbooks (Class 6–12)
- 🌐 Multilingual support (Tamil, Hindi, Telugu, Kannada, Malayalam, etc.)
- 🧠 Conversational memory — resolves pronouns across multiple questions
- 📚 Subjects: Physics, Chemistry, Biology, Mathematics, General
- 🎤 Voice input (Whisper STT)
- 🎓 Difficulty modes: Beginner / Advanced

## Architecture
- **LLM**: `Qwen/Qwen2.5-7B-Instruct`
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector DB**: Chroma Cloud
- **Frontend**: Streamlit
- **Deployment**: Hugging Face Docker Space
