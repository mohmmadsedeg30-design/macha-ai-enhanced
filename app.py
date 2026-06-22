#!/usr/bin/env python3
"""
Macha AI v2.0 - Zero-API-Key Unified Model
English-first | Arabic support | 100% Free
Engines: Local Transformer + Local Pipeline + Ollama + Smart Fallback
"""

import sys, os, json, torch, requests, re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macha.model.transformer import MachaModel, MachaTokenizer
from macha.utils.responses import SmartResponseSystem

app = Flask(__name__)
CORS(app)

# ============ CONFIG ============
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'checkpoints/macha_model.pt'
TOKENIZER_PATH = 'checkpoints/macha_model_tokenizer.json'

# ============ LOAD SYSTEMS ============
model = None
tokenizer = None
response_system = SmartResponseSystem()


def load_local_model():
    """Load Macha local transformer model"""
    global model, tokenizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            tokenizer = MachaTokenizer()
            tokenizer.load(TOKENIZER_PATH)
            model = MachaModel(vocab_size=len(tokenizer.word2idx), device=DEVICE)
            model.load_checkpoint(MODEL_PATH)
            model.eval()
            print(f"[OK] Macha local model loaded on {DEVICE}")
            return True
    except Exception as e:
        print(f"[WARN] Local model error: {e}")
    return False


def is_arabic(text):
    """Detect Arabic text"""
    return bool(re.search(r'[؀-ۿ]', text))


def query_ollama(prompt):
    """Query Ollama local server (if running)"""
    try:
        payload = {
            "model": "llama2",
            "prompt": f"You are Macha AI. Answer concisely in the same language as the user.\n\nUser: {prompt}\nMacha:",
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300}
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get('response', '')
        return None
    except:
        return None


def unified_generate(user_message):
    """
    Unified Pipeline (NO API KEYS NEEDED):
    1. Try Local Macha Model
    2. Try Ollama (if running locally)
    3. Smart Response System (always works)
    """

    # 1. Try local transformer model
    if model and tokenizer:
        try:
            response = model.generate(tokenizer, prompt=user_message, max_new_tokens=60, temperature=0.8)
            response = response.replace(user_message, '').strip()
            if response and len(response) > 5:
                return response, 'macha-local'
        except:
            pass

    # 2. Try Ollama (local LLM, no internet needed after download)
    ollama_response = query_ollama(user_message)
    if ollama_response and len(ollama_response) > 10:
        return ollama_response, 'ollama'

    # 3. Smart fallback (always works, no internet needed)
    return response_system.get_response(user_message), 'macha-fallback'


# Load local model on startup
load_local_model()


# ============ ROUTES ============

@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    response_text, source = unified_generate(user_message)

    return jsonify({
        'response': response_text,
        'source': source,
        'language': 'arabic' if is_arabic(user_message) else 'english',
        'device': DEVICE
    })


@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_input = data.get('input', '')
    correct_output = data.get('correct_output', '')

    if not all([user_input, correct_output]):
        return jsonify({'error': 'Missing data'}), 400

    response_system.learn_response(user_input, correct_output)

    feedback_path = 'data/feedback.json'
    existing = []
    if os.path.exists(feedback_path):
        with open(feedback_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    existing.append({
        'input': user_input,
        'wrong_output': data.get('wrong_output', ''),
        'correct_output': correct_output
    })

    os.makedirs('data', exist_ok=True)
    with open(feedback_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return jsonify({'status': 'success', 'message': 'Learning from feedback!'})


@app.route('/api/stats')
def stats():
    feedback_path = 'data/feedback.json'
    feedback_count = 0
    if os.path.exists(feedback_path):
        with open(feedback_path, 'r', encoding='utf-8') as f:
            feedback_count = len(json.load(f))

    ollama_available = False
    try:
        requests.get("http://localhost:11434", timeout=2)
        ollama_available = True
    except:
        pass

    return jsonify({
        'model_loaded': model is not None,
        'device': DEVICE,
        'feedback_count': feedback_count,
        'knowledge_base_size': len(response_system.knowledge_base),
        'ollama_available': ollama_available,
        'version': '2.0.0',
        'language': 'English-first with Arabic support',
        'api_key_required': False
    })


if __name__ == '__main__':
    print("=" * 50)
    print("   Macha AI v2.0 - Zero API Key")
    print("=" * 50)
    print("\n[OK] Running at: http://localhost:5000")
    print("[OK] Pipeline: Local -> Ollama -> Smart Fallback")
    print("[OK] Languages: English (primary) | Arabic (full support)")
    print("[OK] NO API KEYS NEEDED!")
    print("\n[NOTE] To enhance with Ollama:")
    print("       1. Install: https://ollama.com")
    print("       2. Run: ollama run llama2")
    print("       3. Macha will auto-detect and use it!")
    app.run(host='0.0.0.0', port=5000, debug=True)
