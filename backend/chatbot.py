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

def preprocess_query(query: str) -> dict:
    """
    Uses Mistral to detect language, translate the query to English,
    and classify the subject.
    """
    client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
    
    prompt = f"""<s>[INST] Analyze the following user question: "{query}"
Your task is to:
1. Detect Language (e.g., Tamil, Hindi, English, Tanglish).
2. Translate the query into clear English.
3. Identify Subject (Physics, Chemistry, Biology, Mathematics, General).

Reply STRICTLY in JSON format with exactly these keys: "language", "translated_query", "subject".
Example: {{"language": "English", "translated_query": "What is photosynthesis?", "subject": "Biology"}}
[/INST]"""

    try:
        response = client.text_generation(prompt, max_new_tokens=100, temperature=0.1)
        json_output = response.strip()
        if "```json" in json_output:
            json_output = json_output.split("```json")[-1].split("```")[0].strip()
        elif "```" in json_output:
            json_output = json_output.split("```")[-1].split("```")[0].strip()
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
        return {
            "language": "Unknown",
            "translated_query": query,
            "subject": "General",
        }

def generate_answer(context: str, user_query: str, chat_history: list, difficulty: str, metadata: dict, answer_language: str = "English") -> str:
    """
    Sends the retrieved NCERT context and question to Mistral.
    Answers strictly in the `answer_language` chosen by the user.
    """
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_api_token_here":
        return "Error: HF_TOKEN environment variable not set or invalid."

    try:
        subject = metadata.get("subject", "General")
        
        diff_instruction = (
            "Very simple and basic explanation for younger school students."
            if difficulty == "Beginner"
            else "Detailed and complete explanation with rich scientific information."
        )
        
        # Language instruction — the MOST important rule
        if answer_language and answer_language.lower() != "english":
            lang_instruction = (
                f"⚠️ CRITICAL: You MUST write your ENTIRE response in {answer_language}. "
                f"Every word of your explanation must be in {answer_language}. "
                f"Do NOT mix English words (except for technical/scientific proper nouns like element names, formulas). "
                f"The student has chosen {answer_language} as their preferred language."
            )
        else:
            lang_instruction = "Write your response clearly in English."
            
        system_instruction = (
            f"You are a friendly teacher explaining {subject} concepts to Indian school students.\n"
            "You MUST base your answer ONLY on the provided NCERT textbook context below. "
            "Do NOT add information that is not present in the context.\n"
            f"Difficulty Level: {diff_instruction}\n\n"
            
            f"{lang_instruction}\n\n"
            
            "CRITICAL RULES FOR OUTPUT:\n"
            "1. DO NOT echo or include headers like 'Previous Conversation:', 'Current User Message:', 'Assessment:', or 'Student Question:' in your response.\n"
            "2. Start your explanation directly.\n"
            "3. Base every fact strictly on the NCERT textbook context provided. If the answer is not in the context, say so clearly.\n\n"
            
            "--- STANDARD EXPLANATION STRUCTURE ---\n"
            "If the user asks an educational question, structure your answer as:\n\n"
            "Definition\n[Short definition from the textbook]\n\n"
            "Explanation\n[Simple explanation based on the textbook]\n\n"
            "Examples\n[Real-life examples if mentioned in the textbook]\n\n"
            "Formula / Solving Steps\n[Provide formula or math steps 1,2,3 if applicable, else omit]"
        )

        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-3:]])
        
        user_content = (
            f"Context from Textbooks:\n{context}\n\n"
            f"Chat History Context:\n{history_text}\n\n"
            f"Question:\n{user_query}\n"
        )

        client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
        
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()

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
