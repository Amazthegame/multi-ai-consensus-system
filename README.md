# 🤖 Multi-AI Consensus System with Web Search

A Python tool that queries multiple AI platforms simultaneously (Claude, ChatGPT, Gemini, Perplexity), enables web search where available, and synthesizes their responses into a single consensus answer with aggregated sources.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 What It Does

Instead of manually asking the same question across multiple AI platforms, this tool:

- ✅ **Queries multiple AI models simultaneously** (Claude, GPT-4, Gemini, Perplexity)
- 🔍 **Enables web search** on each platform for current, real-time information
- 📊 **Collects individual responses** with sources and citations
- 🧠 **Synthesizes consensus** using Claude to identify agreements/disagreements
- 📚 **Aggregates sources** from all platforms into one comprehensive list
- 💾 **Saves results** to JSON for future reference

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/multi-ai-consensus-system.git
cd multi-ai-consensus-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Keys

Get API keys from these platforms (all optional - use at least one):

- **Claude** (Recommended): https://console.anthropic.com - $5 free credit
- **Perplexity** (Recommended): https://www.perplexity.ai/settings/api - Free tier available
- **Gemini**: https://ai.google.dev - Free tier
- **OpenAI**: https://platform.openai.com - Paid

Copy `.env.example` to `.env` and add your keys.

## 🚀 Usage
```bash
python multi_ai_consensus.py
```

The script will prompt you for API keys and enter interactive query mode.

## 💡 Use Cases

- **Research**: Get multiple perspectives with sources on complex topics
- **Fact-Checking**: Cross-reference answers across different AI models
- **Current Events**: Leverage web search for up-to-date information
- **Decision Making**: See where models agree/disagree before taking action

## 📝 License

This project is licensed under the MIT License.

## 📧 Contact

Ahmed - your.email@example.com

Project Link: [https://github.com/YOUR-USERNAME/multi-ai-consensus-system](https://github.com/YOUR-USERNAME/multi-ai-consensus-system)

---

⭐ If you found this helpful, please star this repository!
```

Replace `YOUR-USERNAME` and `your.email@example.com` with your actual information.

## 📤 Now Publish to GitHub

1. **Open GitHub Desktop**
2. You should see all 5 files listed on the left
3. At the bottom left, in the "Summary" field, type:
```
   Initial commit: Multi-AI consensus system