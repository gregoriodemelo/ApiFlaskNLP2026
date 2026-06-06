import os
from dotenv import load_dotenv
import google.generativeai as generativeai
import pandas as pd
import numpy as np  
import pickle

load_dotenv()

chave_secreta = os.environ.get('GEMINI_API_KEY', '')
if not chave_secreta:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it with your Gemini API key.")

generativeai.configure(api_key=chave_secreta)

# Link público da sua planilha UBS
csv_url = 'https://docs.google.com/spreadsheets/d/1R69X6fbdmwC9O1atZUo8_PZQYae08SPl/export?format=csv'
df = pd.read_csv(csv_url)
print(df.head())

model = 'models/gemini-embedding-001'

def gerarEmbeddings(title, text):
    result = generativeai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document",
        title=title
    )
    return result['embedding']

def gerarBuscarConsulta(consulta, dataset):
    embedding_consulta = generativeai.embed_content(
        model=model,
        content=consulta,
        task_type="retrieval_query",
    )
    produtos_escalares = np.dot(np.stack(dataset["Embeddings"]), embedding_consulta['embedding'])
    indice = np.argmax(produtos_escalares)
    return dataset.iloc[indice]['Texto']

# Processa linha a linha com contador e tratamento de erro
embeddings = []
for i, row in df.iterrows():
    try:
        emb = gerarEmbeddings(row["Título"], row["Texto"])
        embeddings.append(emb)
        print(f"Linha {i} processada com sucesso")
    except Exception as e:
        print(f"Erro na linha {i}: {e}")
        embeddings.append(None)  # mantém posição mesmo com erro

df["Embeddings"] = embeddings
print(df)

# Salva o dataset com embeddings (mesmo parcial)
pickle.dump(df, open('datasetEmbeddings.pkl','wb'))
