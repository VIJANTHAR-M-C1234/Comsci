import os
import requests
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Supported answer languages
SUPPORTED_LANGUAGES = [
    "English",
    "Tamil",
    "Hindi",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Bengali",
    "Gujarati",
    "Punjabi",
    "Urdu",
]

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Sends the microphone audio to Whisper API and returns text.
    Uses openai/whisper-large-v3-turbo for fast multilingual STT.
    """
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_api_token_here":
        return "Error: HF_TOKEN missing."
        
    api_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/flac"
    }
    try:
        response = requests.post(api_url, headers=headers, data=audio_bytes)
        response.raise_for_status()
        result = response.json()
        return result.get("text", "Error processing audio")
    except Exception as e:
        return f"Audio Error: {str(e)}"

# ── New HF Inference Router (replaces api-inference.huggingface.co) ──
HF_API_BASE = "https://router.huggingface.co/hf-inference/models"
MODEL_ID    = "google/gemma-2-2b-it"
HF_API_URL  = f"{HF_API_BASE}/{MODEL_ID}"


def _call_hf_api(prompt: str, max_new_tokens: int = 800, temperature: float = 0.2) -> str:
    """
    Direct POST to legacy HF Inference API.
    Bypasses InferenceClient provider routing — works on ALL free HF accounts.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "do_sample": True,
        },
        "options": {"wait_for_model": True},
    }
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()
    raise ValueError(f"Unexpected API response: {result}")


def preprocess_query(query: str) -> dict:
    """
    Detects language, translates to English, classifies subject.
    Uses direct HF API POST — no provider needed.
    """
    prompt = (
        "<start_of_turn>user\n"
        "You are a language detection assistant. "
        "Analyze this question: \"" + query + "\"\n"
        "Reply ONLY with a valid JSON object with exactly these keys: \"language\", \"translated_query\", \"subject\". "
        "Subject must be one of: Physics, Chemistry, Biology, Mathematics, General. "
        "Example: {\"language\": \"English\", \"translated_query\": \"What is photosynthesis?\", \"subject\": \"Biology\"}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    try:
        json_output = _call_hf_api(prompt, max_new_tokens=120, temperature=0.1)
        if "```json" in json_output:
            json_output = json_output.split("```json")[-1].split("```")[0].strip()
        elif "```" in json_output:
            json_output = json_output.split("```")[1].strip()
        start = json_output.find("{")
        end = json_output.rfind("}") + 1
        if start != -1 and end > start:
            json_output = json_output[start:end]
        data = json.loads(json_output)
        return {
            "language": data.get("language", "English"),
            "translated_query": data.get("translated_query", query),
            "subject": data.get("subject", "General"),
        }
    except Exception:
        return {"language": "Unknown", "translated_query": query, "subject": "General"}


def generate_answer(context: str, user_query: str, chat_history: list, difficulty: str, metadata: dict, answer_language: str = "English") -> str:
    """
    Generates NCERT answer via direct HF Inference API POST.
    No provider router — works on all free HF accounts.
    """
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_api_token_here":
        return "Error: HF_TOKEN not set."

    try:
        subject = metadata.get("subject", "General")

        diff_instruction = (
            "Use very simple language for younger school students."
            if difficulty == "Beginner"
            else "Give a detailed explanation with scientific depth."
        )

        lang_instruction = (
            f"IMPORTANT: Write your ENTIRE answer in {answer_language}. "
            "Only technical terms and formulas may stay in English."
            if answer_language and answer_language.lower() != "english"
            else "Write your response in clear English."
        )

        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history[-3:]
        ])

        # Gemma-2 Chat Format — works on new HF Inference Router
        full_prompt = (
            "<start_of_turn>user\n"
            "You are a friendly NCERT teacher for Indian school students.\n"
            f"Subject: {subject}. {diff_instruction}\n"
            f"{lang_instruction}\n\n"
            "RULES:\n"
            "- Answer ONLY from the NCERT context provided.\n"
            "- Start directly. Do not repeat the question or add chat headers.\n"
            "- Structure: Definition -> Explanation -> Examples -> Formula (if needed).\n"
            "- If the answer is not in the context, say so clearly.\n\n"
            f"NCERT Context:\n{context}\n\n"
            f"Recent Chat History:\n{history_text}\n\n"
            f"Question: {user_query}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        return _call_hf_api(full_prompt, max_new_tokens=800, temperature=0.2)

    except Exception as e:
        return f"API Connection Error: {str(e)}"




def ask_chatbot(question: str, retriever, chat_history: list, difficulty: str, answer_language: str = "English") -> dict:
    """
    RAG pipeline: detects subject/translates -> retrieves from vector DB -> generates answer.
    Returns a dict containing response and metadata.
    answer_language: the language the model should reply in (user's choice).
    """
    try:
        # Preprocess the query to translate and find subject
        metadata = preprocess_query(question)
        translated_q = metadata["translated_query"]
        
        # Store the chosen answer language in metadata for display
        metadata["answer_language"] = answer_language
        
        # 1. Retrieve relevant NCERT chunks based on translated English question
        docs = retriever.invoke(translated_q)
        
        if not docs:
            response = "I could not find relevant information in the NCERT textbook. Please rephrase your question or check your textbook directly."
            return {"response": response, "metadata": metadata}
            
        # 2. Extract context from retrieved documents
        context = "\n\n".join([f"Document Part:\n{doc.page_content}" for doc in docs])
        
        # 3. Generate answer strictly from the NCERT vector DB context
        response_text = generate_answer(context, question, chat_history, difficulty, metadata, answer_language)
        return {"response": response_text, "metadata": metadata}
        
    except Exception as e:
        return {"response": f"System Error Occurred: {str(e)}", "metadata": {}}
