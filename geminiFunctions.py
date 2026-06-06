import google.generativeai as generativeai
from google import genai
from google.genai import types
import numpy as np
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Modelo de embeddings
model = 'models/gemini-embedding-001'

def gerarBuscarConsulta(consulta, dataset):
    try:
        # Gera embedding da consulta
        embedding_consulta = generativeai.embed_content(
            model=model,
            content=consulta,
            task_type="retrieval_query",
        )
    except Exception as e:
        # Tratamento de erro 429 ou outros
        if "ResourceExhausted" in str(e) or "429" in str(e):
            return "O serviço de embeddings está temporariamente indisponível (erro 429). Tente novamente em alguns minutos."
        return f"Erro ao gerar embedding da consulta: {str(e)}"

    # Filtra apenas embeddings válidos e com shape correto
    embeddings_validos = [
        emb for emb in dataset["Embeddings"]
        if emb is not None and len(emb) == len(embedding_consulta['embedding'])
    ]

    if not embeddings_validos:
        return "Não foi possível encontrar contexto válido para a consulta."

    # Calcula produto escalar apenas com embeddings válidos
    produtos_escalares = np.dot(np.stack(embeddings_validos), embedding_consulta['embedding'])

    # Encontra índice do embedding mais próximo
    indice = np.argmax(produtos_escalares)

    # Retorna o texto correspondente
    return dataset.iloc[indice]['Texto']

# Modelo para melhorar a resposta
modelo = 'gemini-3-flash-preview'

def melhorarResposta(inputText):
    try:
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        model = modelo
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=inputText),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""
                Você é um assistente baseado em RAG (Retrieval-Augmented Generation).
                Utilize exclusivamente o conteúdo recuperado da base de conhecimento para responder à consulta do usuário.
                Considere a consulta e o contexto recuperado, e gere uma resposta clara, objetiva e coerente, reescrevendo as informações de forma natural sem copiar literalmente o texto original.
                Não invente informações que não estejam presentes no contexto fornecido e não apresente múltiplas opções de resposta.
                """),
            ],
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        return response.text

    except Exception as e:
        if "ResourceExhausted" in str(e) or "429" in str(e):
            return "O serviço de geração de respostas está temporariamente indisponível (erro 429). Tente novamente em alguns minutos."
        return f"Erro ao gerar resposta: {str(e)}"
