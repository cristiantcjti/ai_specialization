import openai

client = openai.OpenAI(base_url="https://api.groq.com/openai/v1")

response = client.responses.create(
    model = "llama-3.1-8b-instant", 
    instructions="Answer in a simple way in one paragraph.",
    input="What is machine learing?"
)

print(response.output)
print(response.output_text)

