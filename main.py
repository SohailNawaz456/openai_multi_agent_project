import os 
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, set_tracing_disabled
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

def main():
    print("🎓 Welcome To Math Tutor CLI ✨")   # Added emojis here
    user_question = input("❓ What is your question? ")

    agent = Agent(
        name="Math Tutor",
        instructions="you provide help with math questions.",
        model=model
    )
    result = Runner.run_sync(agent, user_question)
    rich.print("✅ Answer:", result.final_output)           # Added emojis here

if __name__ == "__main__":
    main()
