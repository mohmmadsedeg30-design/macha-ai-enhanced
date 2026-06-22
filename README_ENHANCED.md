# 🍵 Macha AI v2.0 - Enhanced Edition

**AI Chat Platform with Zero API Keys | English & Arabic Support | 100% Free**

---

## ✨ Features

- 🧠 **Unified AI Pipeline**: Local Model → Ollama → Smart Fallback
- 🌍 **Bilingual Support**: Full English & Arabic support with auto-detection
- 💬 **Web Chat Interface**: Beautiful, responsive UI with gradient animations
- 🖥️ **CLI Mode**: Direct terminal chat for Termux & Linux
- 📚 **Learning System**: Learns from user corrections and feedback
- ⚡ **Zero Dependencies**: No API keys needed, works offline
- 📱 **Mobile Friendly**: Responsive design for all devices
- 🎨 **Modern UI**: Gradient animations, smooth transitions, emoji support

---

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
# Clone repository
git clone https://github.com/mohmmadsedeg30-design/macha-ai-enhanced.git
cd macha-ai-enhanced

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# Open in browser
# http://localhost:5000
```

### Option 2: CLI Mode (Termux / Terminal)

```bash
# Run in CLI mode
python app.py --cli

# Or with custom port
python app.py --port 8000
```

---

## 📱 Termux Installation (Android)

```bash
# Update packages
pkg update && pkg upgrade -y

# Install Python and dependencies
pkg install python python-pip clang make -y

# Clone repository
git clone https://github.com/mohmmadsedeg30-design/macha-ai-enhanced.git
cd macha-ai-enhanced

# Install requirements
pip install -r requirements.txt

# Run CLI mode
python app.py --cli

# Or run web server
python app.py
# Then open http://192.168.x.x:5000 on another device
```

---

## 🎮 Usage

### Web Chat Interface

1. Open `http://localhost:5000` in your browser
2. Type messages in English or العربية
3. Click the send button (➤) or press Enter
4. Rate responses with ✓ Correct or ✗ Wrong buttons
5. The AI learns from your feedback!

### CLI Mode

```bash
python app.py --cli

# Interactive chat:
👤 You: What is machine learning?
🧠 Macha is thinking...
💬 Macha: Machine learning is a subset of artificial intelligence...
   └─ Source: macha-fallback

# Commands:
# - Type 'exit' or 'quit' to leave
# - Type 'stats' to see system info
```

### API Endpoints

```bash
# Chat endpoint
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'

# Response:
# {
#   "response": "Hello! I'm Macha...",
#   "source": "macha-fallback",
#   "language": "english",
#   "device": "cpu"
# }

# Feedback endpoint
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
#     "input": "What is AI?",
#     "wrong_output": "AI is nothing",
#     "correct_output": "AI is artificial intelligence"
#   }'

# Stats endpoint
curl http://localhost:5000/api/stats
```

---

## 🧠 AI Pipeline

### 1. Local Macha Model
- Custom transformer trained on Q&A data
- Runs on CPU/GPU
- Fastest response time

### 2. Ollama Integration
- Download: https://ollama.com
- Run: `ollama run llama2`
- Macha auto-detects and uses it

### 3. Smart Fallback
- Rule-based response system
- Always works, even offline
- Learns from user corrections

---

## 🔧 Configuration

### Custom Port

```bash
python app.py --port 8080
```

### Custom Host

```bash
python app.py --host 0.0.0.0 --port 5000
```

### Enable Ollama

1. Install Ollama: https://ollama.com
2. Run: `ollama run llama2`
3. Macha will auto-detect on http://localhost:11434

---

## 📊 Testing

Run the included test suite:

```bash
node test_frontend.js
```

Tests:
- ✅ Server connection
- ✅ English chat
- ✅ Arabic chat
- ✅ Stats API
- ✅ Feedback system

---

## 📁 Project Structure

```
macha-ai-enhanced/
├── app.py                 # Main Flask server
├── templates/
│   └── chat.html         # Web UI (HTML/CSS/JS)
├── macha/
│   ├── model/
│   │   ├── transformer.py # AI model
│   │   └── tokenizer.py   # Text tokenizer
│   └── utils/
│       └── responses.py   # Response system
├── data/
│   └── feedback.json     # User feedback log
├── requirements.txt      # Python dependencies
└── test_frontend.js      # Test suite
```

---

## 🌐 Browser Support

- ✅ Chrome/Chromium
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Mobile browsers

---

## 🎨 UI Features

### Responsive Design
- Desktop: Full-width chat interface
- Tablet: Optimized layout
- Mobile: Touch-friendly buttons

### Animations
- Gradient background shift
- Message slide-in animations
- Typing indicator dots
- Button hover effects
- Smooth scrolling

### Accessibility
- Keyboard navigation (Enter to send)
- High contrast text
- Emoji support
- RTL Arabic text support

---

## 🔒 Privacy & Security

- ✅ No data sent to external servers
- ✅ All processing local or via Ollama
- ✅ Feedback stored locally in `data/feedback.json`
- ✅ No tracking or analytics
- ✅ Open source (MIT License)

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Or use different port
python app.py --port 8000
```

### Ollama Not Detected
```bash
# Check if Ollama is running
curl http://localhost:11434

# If not running, install and start:
# https://ollama.com
ollama run llama2
```

### Torch Not Found (Termux)
```bash
# Use pkg instead of pip
pkg install python-torch
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Response Time | 0.5-2s (fallback), 1-5s (Ollama) |
| Memory Usage | ~200MB (minimal) |
| CPU Usage | Low (optimized) |
| Offline Support | ✅ Yes |
| Mobile Support | ✅ Yes |

---

## 🤝 Contributing

Found a bug or have a feature request? 
- Open an issue on GitHub
- Submit a pull request
- Share feedback via the chat interface

---

## 📄 License

MIT License - Free for personal and commercial use

---

## 🙏 Credits

- Built with Flask, PyTorch, and ❤️
- Inspired by modern AI chat interfaces
- Special thanks to the open-source community

---

## 📞 Support

- 📧 Email: support@macha-ai.com
- 🐦 Twitter: @MachaAI
- 💬 Discord: [Join our community]
- 📖 Docs: https://docs.macha-ai.com

---

## 🎯 Roadmap

- [ ] Voice input/output
- [ ] Multi-language support (10+ languages)
- [ ] User accounts & chat history
- [ ] Mobile app (iOS/Android)
- [ ] API for third-party integration
- [ ] Advanced model fine-tuning
- [ ] Real-time collaboration

---

**Made with 🍵 by the Macha AI Team**

*Last Updated: June 22, 2026*
