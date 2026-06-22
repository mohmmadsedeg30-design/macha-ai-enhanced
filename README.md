# Macha AI v2.0

<div align="center">

# Macha AI
### Unified AI Model | English-First | Arabic Support

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## What is Macha?

**Macha** is an open-source AI model that combines multiple free AI engines into one unified system:

- **Local Transformer** - Small model that learns from you
- **HuggingFace** - Free API access to powerful models (Gemma, Mistral, Zephyr, Phi)
- **Ollama** - Run LLMs locally (Llama2, Mistral, etc.)
- **Multilingual** - English (primary) + Arabic (full support via AraGPT2)

## Quick Start

```bash
# 1. Extract
unzip macha-ai.zip
cd macha-ai

# 2. Install
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000
```

## How It Works

Macha automatically selects the best available engine:

```
User Input -> Detect Language -> Try Local Model -> Try HuggingFace -> Try Ollama -> Smart Fallback
```

**No API keys needed!** HuggingFace free tier works without authentication for public models.

## Languages

- **English** - Primary language, fully optimized
- **Arabic** - Full support via AraGPT2 and multilingual models

## Features

- Beautiful gradient UI with animations
- Learns from mistakes (feedback buttons)
- Code syntax highlighting
- Auto language detection
- Source tag shows which engine answered
- Mobile responsive

## Enhance with Ollama (Optional)

```bash
# Install Ollama: https://ollama.com
ollama run llama2
# Now Macha will use local LLaMA for better responses!
```

## License

MIT License - Open Source
