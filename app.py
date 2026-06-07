from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as generativeai
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os
from geminiFunctions import gerarBuscarConsulta, melhorarResposta

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar Flask
app = Flask(__name__)
CORS(app)  # habilita CORS para toda a aplicação

# Configurações do modelo e embeddings
modelo = 'gemini-3-flash-preview'
modeloEmbeddings = pickle.load(open('datasetEmbeddings.pkl', 'rb'))
chave_secreta = os.getenv('GEMINI_API_KEY')
generativeai.configure(api_key=chave_secreta)

# Endpoint raiz
@app.route("/")
def home():
    consulta = "Quem é você ?"
    resposta = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resposta}"
    response = melhorarResposta(prompt)
    return response

# Endpoint de API
@app.route("/api", methods=["POST"])
def results():
    # Verifica a chave de autorização
    auth_key = request.headers.get("Authorization")
    print("Authorization recebido:", repr(auth_key))  # <-- loga exatamente o que chegou

    if not auth_key or auth_key.strip() != chave_secreta.strip():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(force=True)
    consulta = data["consulta"]
    resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resultado}"
    response = melhorarResposta(prompt)
    return jsonify({"mensagem": response})

# Bloco principal para rodar localmente
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
