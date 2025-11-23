from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages = [
        {
            "role": "system", 
            "content": "Atue como um especialista em machine learning"
        },
        {
            "role": "user", 
            "content": "De forma simple, o que é machine learning?"
        }
    ],
    temperature=0.5,
    top_p=0.8
)

print(response.choices[0].message.content)