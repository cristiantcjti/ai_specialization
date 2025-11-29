from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq()

# qdrant = QdrantClient(":memory:")
qdrant = QdrantClient(path="db/data")

vector_size = model.get_sentence_embedding_dimension()

if vector_size is None:
    msg = "Failed to get embedding dimension from model"
    raise ValueError(msg)

qdrant.create_collection(
    collection_name="ml_documents",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

points = []
for index, document in enumerate(documents):
    embedding = model.encode(document).tolist()
    points.append(
        PointStruct(id=index, vector=embedding, payload={"text": document})
    )

qdrant.upsert(collection_name="ml_documents", points=points, wait=True)


def retrieve(query: str, top_k: int = 3) -> list[tuple[str, float]]:
    query_embedding = model.encode(query).tolist()
    search_result = qdrant.query_points(
        collection_name="ml_documents",
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    if search_result.points is None:
        return []

    return [
        (result.payload["text"], result.score)
        for result in search_result.points
        if result.payload is not None and result.score is not None
    ]


def generate_answer(
    query: str, retrieve_docs: list[tuple[str, float]]
) -> str | None:
    contexto = "\n".join([doc for doc, _ in retrieve_docs])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Você é um especialista em machine learning. Use apenas o contexto fornecido para responder as perguntas.",
            },
            {
                "role": "user",
                "content": f"Contexto:\n{contexto}\n\nPergunta: {query}",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def rag(
    query: str, top_k: int = 3
) -> tuple[str | None, list[tuple[str, float]]]:
    retrieved = retrieve(query=query, top_k=top_k)
    answer = generate_answer(query=query, retrieve_docs=retrieved)
    return answer, retrieved


answer, docs = rag("O que é machine learning?")

print(answer)
for doc, similarity in docs:
    print(f"- {similarity:.3f}: {doc}")
