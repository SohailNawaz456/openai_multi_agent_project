import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, set_tracing_disabled
import rich

# Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)

# Fetch Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise Exception("GEMINI_API_KEY is not set in .env file")

# Setup OpenAI/Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Setup model and configuration
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define FAQ Agent
faq_agent = Agent(
    name="FAQ Bot",
    instructions="You are a helpful FAQ bot. Answer simple questions like 'What is your name?', 'What can you do?', etc.",
    model=model
)

# Main loop for interaction
def main():
    print("\n🤖 Ask a question (type 'exit' to quit):")
    while True:
        user_question = input("❓ You: ")
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        result = Runner.run_sync(faq_agent, user_question)
        rich.print("✅ [bold green]Bot:[/]", result.final_output)

if __name__ == "__main__":
    main()
