import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, set_tracing_disabled, function_tool
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
def multiplication(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

def main():
    print("🎓 Welcome To Math Function Tool Agent ✨")

    agent = Agent(
        name="Math Function Tool Agent",
        instructions="You are a math function tool agent. Use the multiplication function when needed.",
        model=model,
        tools=[multiplication]
    )

    while True:
        user_question = input("❓ What is your question? ")
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