# 🤖 Multi-Tool AI Agents

This project showcases a set of intelligent Python agents built using OpenAI SDK and tool functions. Each agent is designed to perform a specific task, such as answering FAQs, solving math problems, fetching real-time weather data, and handling multi-tool queries.

## 📁 Project Structure

multi-tool-ai-agents/
├── faq_agent.py # Answers predefined frequently asked questions
├── math_agent.py # Solves basic math problems using tools
├── weather_agent.py # Fetches live weather data using WeatherAPI
├── multi_agent.py # Combines math and weather tools into one agent
├── .env # Stores API keys (not pushed to GitHub)
├── README.md # Project documentation


## 🧠 Agents Overview

### 1. 📚 `faq_agent`
- Uses a predefined list of frequently asked questions and responses.
- Built for static responses, ideal for customer support demos.

### 2. ➕ `math_agent`
- Performs addition (can be extended to other operations).
- Uses tool functions to dynamically compute answers from user input.

### 3. 🌤️ `weather_agent`
- Fetches real-time weather from [WeatherAPI](https://www.weatherapi.com/).
- Smart emoji + condition mapping based on temperature and forecast.

### 4. 🧰 `multi_agent`
- Combines multiple tools (e.g., `get_weather` and `add`) in one agent.
- Automatically selects the appropriate tool based on the user's question.

## 🛠️ Tech Stack

- Python 3.10+
- [OpenAI SDK (Unofficial)](https://github.com/openai/openai-python)
- `dotenv` for secure API key handling
- `requests` for external API communication
- `rich` for better CLI output

## 🔐 Environment Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key_here
WEATHER_API_KEY=your_weatherapi_key_here

▶️ How to Run
git clone https://github.com/your-username/multi-tool-ai-agents.git
cd multi-tool-ai-agents

nstall requirements:
pip install -r requirements.txt

Run an agent:
python weather_agent.py

📌 Future Improvements
Add subtraction, multiplication, and division tools

Enhance FAQ agent with semantic search

Use LangChain or LangGraph for better agent orchestration

📄 License
This project is for educational purposes.

Feel free to fork, modify, or improve. If you like it, star it! ⭐


---

### ✅ Optional:
Let me know if you want:
- A `requirements.txt`
- Example screenshots or GIFs in the README
- Urdu version of README (if needed for local submission)

Want me to generate the `requirements.txt` for you based on your agents?

