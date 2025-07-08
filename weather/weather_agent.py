import os 
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, function_tool, set_tracing_disabled
import rich

load_dotenv()
set_tracing_disabled(disabled=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise Exception("GEMINI_API_KEY is not set in .env file")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def get_weather(city: str) -> str:
    """Returns the weather for a given city."""
    return f"The weather in {city} is 31 C. with clear skies and a gentle breeze. Perfect for a day out! 🌤️"

def main():
    print("🌤️ Welcome to the Weather Agent! 🌤️")
    print("Ask about the weather in any city (type 'exit' to quit):")
    
    agent = Agent(
        name="Weather Agent",
        instructions="You are a weather agent. Use the get_weather function to provide weather information.",
        model=model,
        tools=[get_weather]
    )
    while True:
        user_question = input("❓ What is your question? ")
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

