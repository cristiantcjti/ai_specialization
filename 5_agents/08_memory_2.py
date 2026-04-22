import os

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "openai_base_url": "https://api.groq.com/openai/v1",
            "api_key": os.environ["GROQ_API_KEY"],
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "memory",
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "embedding_model_dims": 384,
        },
    },
    "embedder": {
        "provider": "fastembed",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
}

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

memory = Memory.from_config(config_dict=config)


def chat_with_memories(message: str, user_id: str = "cris") -> str:
    relevant_memories = memory.search(
        query=message, filters={"user_id": user_id}, limit=3
    )
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
    )

    input_prompt = f""" You are a personal assistent.
    Answer the question taking into consideration the user's memories.
    User memories: {memories_str}
    Question: {message}
    """

    response = client.responses.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        input=input_prompt,
    )

    assistant_response = response.output_text

    if not assistant_response:
        raise ValueError("Failed to get response.")

    messages = [
        {
            "role": "user",
            "content": message,
        },
        {
            "role": "assistant",
            "content": assistant_response,
        },
    ]

    memory.add(messages=messages, user_id=user_id)

    return assistant_response


def main() -> None:
    print("Chat with AI (Type exit to end the chat)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bye!")
            break
        print(f"AI: {chat_with_memories(message=user_input)}")


if __name__ == "__main__":
    main()
