from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import os
import requests
import rich

# Load environment variables
load_dotenv()
from dotenv import load_dotenv
import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your environment.")

# Reference: https://ai.google.dev/gemini-api/docs/openai

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True

)

# Weather tool using WeatherAPI
@function_tool
def get_weather(city: str) -> str:
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "❌ Weather API key is missing."

    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        temp_c = data["current"]["temp_c"]
        orig_condition = data["current"]["condition"]["text"]

        # Smart condition + emoji mapping
        if temp_c <= 15:
            condition = "Cloudy"
            emoji = "☁️"
        elif 15 < temp_c <= 25:
            condition = "Sunny"
            emoji = "☀️"
        elif 25 < temp_c <= 35:
            condition = "Hot Sunny"
            emoji = "🔥☀️"
        else:
            condition = "Extreme Heat"
            emoji = "🥵🔥"

        orig_condition_lower = orig_condition.lower()
        if "rain" in orig_condition_lower:
            condition = "Rainy"
            emoji = "🌧️"
        elif "snow" in orig_condition_lower:
            condition = "Snowy"
            emoji = "❄️"
        elif "thunder" in orig_condition_lower:
            condition = "Thunderstorm"
            emoji = "⛈️"
        elif "fog" in orig_condition_lower or "mist" in orig_condition_lower:
            condition = "Foggy"
            emoji = "🌫️"
        elif "wind" in orig_condition_lower:
            condition = "Windy"
            emoji = "🌪️"
        elif "cloud" in orig_condition_lower and temp_c > 15:
            condition = "Partly Cloudy"
            emoji = "🌤️"

        return f"✅ The current weather in {city} is {temp_c}°C 🌡️ and {condition} {emoji}"

    except Exception as e:
        return f"❌ Error fetching weather data: {e}"

# Setup the weather agent
agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful assistant who provides real-time weather information using the get_weather tool.",
    tools=[get_weather]
)

# Main user interaction loop
def main():
    print("🌍 Real-Time Weather Agent")
    print("Ask about the weather in any city. Type 'exit' to quit.\n")

    while True:
        user_input = input("❓ Enter your weather question: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        if not user_input.strip():
            rich.print("[yellow]⚠️ Please enter a valid question.[/yellow]")
            continue

        try:
            result = Runner.run_sync(agent, user_input, run_config=config)
            rich.print("✅ Answer:", result.final_output)
        except Exception as e:
            rich.print(f"[red]❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()
