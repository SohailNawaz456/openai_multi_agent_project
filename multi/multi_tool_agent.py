import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    RunConfig,
    function_tool,
    set_tracing_disabled
)
import rich

# Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise Exception("GEMINI_API_KEY is not set in .env file")

# Setup Gemini Client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Setup model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Tool 1: Get weather
@function_tool
def get_weather(city: str) -> str:
    """Returns the weather for a given city."""
    return f"The weather in {city} is 31°C with clear skies and a gentle breeze. 🌤️"

# Tool 2: Add two numbers
@function_tool
def add(a: float, b: float) -> float:
    """Returns the sum of two numbers."""
    return a + b

# Main interactive loop
def main():
    print("🤖 Welcome to the Multi-Tool Agent!")
    print("🌟Example: For ask any weather question or any math question.):")
    
    agent = Agent(
        name="MultiTool Agent",
        instructions="You're a helpful assistant. Use get_weather for weather queries and add for math questions.",
        model=model,
        tools=[get_weather, add]
    )
    
    while True:
        user_question = input("❓ Your question: ")
        if user_question.strip() == "":
            rich.print("[yellow]⚠️ Please enter a question or type 'exit' to quit.[/yellow]")
            continue
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        try:
            result = Runner.run_sync(agent, user_question)
            rich.print("✅ Answer:", result.final_output)
        except Exception as e:
            rich.print(f"[red]❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()
