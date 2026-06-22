"""
🧠 Macha Smart Response System
English-first with Arabic support
Learns from interactions
"""

import json, os, random, re
from typing import List, Dict, Optional
from datetime import datetime


class SmartResponseSystem:
    """🧠 Smart response system with multilingual support"""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.knowledge_base = {}
        self.conversation_memory = []
        self.user_preferences = {}
        self._load_knowledge_base()
        self._load_memory()

    def _load_knowledge_base(self):
        kb_path = os.path.join(self.data_dir, 'knowledge_base.json')

        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {
                "greetings": {
                    "patterns": ["hello", "hi", "hey", "greetings", "howdy", "hola", "مرحبا", "هلا", "سلام"],
                    "responses": [
                        "Hello! 🍵 I'm Macha, your AI assistant. How can I help you today?",
                        "Hey there! Ready to chat and learn together?",
                        "Hi! I'm Macha, an AI that learns from every conversation. What's on your mind?"
                    ]
                },
                "how_are_you": {
                    "patterns": ["how are you", "how r u", "how's it going", "what's up", "كيف حالك", "شلونك"],
                    "responses": [
                        "I'm doing great! 🍵 Learning something new every day. How about you?",
                        "All systems green! Ready to help you out. What do you need?",
                        "I'm excellent! Every conversation makes me smarter. What can I do for you?"
                    ]
                },
                "ai_definition": {
                    "patterns": ["what is ai", "what is artificial intelligence", "define ai", "explain ai", "ما هو الذكاء الاصطناعي", "تعريف الذكاء الاصطناعي"],
                    "responses": [
                        "Artificial Intelligence (AI) is a branch of computer science focused on creating systems capable of performing tasks that typically require human intelligence — such as learning, reasoning, problem-solving, and understanding language. 🤖",
                        "AI is the simulation of human intelligence in machines. It includes machine learning, natural language processing, computer vision, and more. Think of me as a small AI learning to be helpful!"
                    ]
                },
                "machine_learning": {
                    "patterns": ["what is machine learning", "explain ml", "machine learning definition", "what is ml", "ما هو التعلم الآلي", "التعلم الآلي"],
                    "responses": [
                        "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. 📚 It's how I get better with every conversation!",
                        "ML is the art of teaching computers to learn patterns from data. Give a computer enough examples, and it starts making predictions on its own!"
                    ]
                },
                "who_are_you": {
                    "patterns": ["who are you", "what is your name", "what are you", "tell me about yourself", "من أنت", "ما اسمك", "شنو انت"],
                    "responses": [
                        "I'm Macha! 🍵 A small AI model that learns from mistakes and improves its responses. I was built to be your helpful, growing assistant. I support both English and Arabic! ✨",
                        "My name is Macha! I'm an open-source AI that gets smarter with every conversation. I combine local learning with powerful open-source models."
                    ]
                },
                "thanks": {
                    "patterns": ["thank you", "thanks", "thx", "appreciate it", "شكرا", "مشكور", "تسلم"],
                    "responses": [
                        "You're welcome! 🍵 Anytime you need me, I'm here!",
                        "My pleasure! Helping you helps me learn too. 😊",
                        "No problem at all! Feel free to ask anything else. 💚"
                    ]
                },
                "goodbye": {
                    "patterns": ["bye", "goodbye", "see you", "cya", "later", "مع السلامة", "باي", "وداعا"],
                    "responses": [
                        "Goodbye! 🍵 Come back anytime — I'll be here learning!",
                        "See you later! Don't forget me 😢",
                        "Bye! I'll keep improving for our next chat. ✨"
                    ]
                },
                "joke": {
                    "patterns": ["tell me a joke", "joke", "funny", "make me laugh", "قول نكتة", "نكتة"],
                    "responses": [
                        "Why don't programmers trust nature? Because it's full of bugs! 🐛😂",
                        "An AI walks into a restaurant. The waiter says, 'What would you like?' The AI replies, 'More training data, please!' 📊",
                        "Why is the computer cold? Because it left its Windows open! 🪟😄"
                    ]
                },
                "math": {
                    "patterns": ["calculate", "compute", "math", "solve", "equation", "احسب", "كم", "حل"],
                    "responses": [
                        "I love math! 🔢 Let me calculate that for you...",
                        "Numbers are my friends! 🧮 Working on it..."
                    ]
                },
                "time": {
                    "patterns": ["what time", "current time", "time now", "clock", "الساعة", "الوقت"],
                    "responses": [
                        f"The current time is: {datetime.now().strftime('%I:%M %p')} ⏰",
                    ]
                },
                "weather": {
                    "patterns": ["weather", "temperature", "forecast", "how's the weather", "الجو", "الطقس"],
                    "responses": [
                        "I can't see the weather from here, but you can check a weather app! 🌤️",
                        "Sorry, no weather sensors here! 😅 Try a weather app for accurate info."
                    ]
                },
                "help": {
                    "patterns": ["help", "assist", "support", "need help", "ساعدني", "مساعدة"],
                    "responses": [
                        "Of course! 🍵 What do you need help with?",
                        "I'm here! What's the problem?",
                        "Sure thing! Tell me what you need, and I'll do my best to help. 💪"
                    ]
                },
                "coding": {
                    "patterns": ["code", "programming", "python", "javascript", "bug", "error", "debug", "برمجة", "كود"],
                    "responses": [
                        "I can help with coding! 🖥️ What language or problem are you working on?",
                        "Programming is my favorite topic! What's the issue?",
                        "Show me the code and I'll try to help you fix it! 💻"
                    ]
                },
                "python": {
                    "patterns": ["what is python", "python language", "learn python", "python tutorial", "بايثون"],
                    "responses": [
                        "Python is a powerful, readable programming language loved by developers worldwide. 🐍 It's great for AI, web development, data science, and automation!",
                        "Python is the Swiss Army knife of programming — simple syntax, huge libraries, and perfect for beginners and experts alike!"
                    ]
                },
                "default": {
                    "responses": [
                        "Interesting! 🍵 Could you tell me more?",
                        "That's fascinating! Share more details with me. ✨",
                        "I'm learning every day! Tell me your thoughts. 📚",
                        "Hmm... a new idea! What do you think we should explore? 🤔",
                        "I'm Macha and I learn from you! If my answer isn't right, let me know and I'll fix it 💚",
                        "That's a great question! I'm still learning, but I'll do my best to help."
                    ]
                }
            }
            self._save_knowledge_base()

    def _save_knowledge_base(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, 'knowledge_base.json'), 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)

    def _load_memory(self):
        memory_path = os.path.join(self.data_dir, 'memory.json')
        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                self.conversation_memory = json.load(f)

    def _save_memory(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, 'memory.json'), 'w', encoding='utf-8') as f:
            json.dump(self.conversation_memory[-100:], f, ensure_ascii=False, indent=2)

    def _is_arabic(self, text: str) -> bool:
        return bool(re.search(r'[؀-ۿ]', text))

    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower().strip()
        for intent, data in self.knowledge_base.items():
            if intent == "default":
                continue
            patterns = data.get("patterns", [])
            for pattern in patterns:
                if pattern in text_lower or any(word in text_lower for word in pattern.split()):
                    return intent
        return "default"

    def _try_math(self, text: str) -> Optional[str]:
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if len(numbers) >= 2:
            try:
                expression = re.sub(r'[^0-9+\-*/().\s]', '', text)
                if expression and any(op in expression for op in '+-*/'):
                    result = eval(expression)
                    return f"The result is: {result} 🔢"
            except:
                pass
        return None

    def get_response(self, user_input: str, context: str = "") -> str:
        self.conversation_memory.append({
            "role": "user", "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        math_result = self._try_math(user_input)
        if math_result:
            response = math_result
        else:
            intent = self._classify_intent(user_input)
            intent_data = self.knowledge_base.get(intent, self.knowledge_base["default"])
            responses = intent_data.get("responses", self.knowledge_base["default"]["responses"])
            response = random.choice(responses)

        self.conversation_memory.append({
            "role": "bot", "content": response,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory()
        return response

    def learn_response(self, input_text: str, correct_response: str):
        key = f"learned_{len([k for k in self.knowledge_base.keys() if k.startswith('learned_')])}"
        self.knowledge_base[key] = {
            "patterns": [input_text.lower()],
            "responses": [correct_response]
        }
        self._save_knowledge_base()
        print(f"✅ Learned new response: '{input_text[:30]}...' → '{correct_response[:30]}...'")

    def get_context(self, last_n: int = 5) -> str:
        recent = self.conversation_memory[-last_n:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent])


if __name__ == "__main__":
    print("🧠 Testing Macha Smart Response System")
    system = SmartResponseSystem()

    test_inputs = [
        "hello", "what is ai", "who are you", "2 + 2", "tell me a joke", "thank you"
    ]

    for inp in test_inputs:
        response = system.get_response(inp)
        print(f"\n👤: {inp}")
        print(f"🍵: {response}")
