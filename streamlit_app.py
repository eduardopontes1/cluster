import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Qual é o seu grupo?", layout="centered")

st.title("🔍 Descubra seu grupo com a mágica da estatística!")
st.write("Responda algumas perguntas simples e veja com quem você se parece 👀")

# --- PERGUNTAS ---
q1 = st.radio("🧠 Você aprende melhor:", ["Lendo textos", "Vendo vídeos"])
q2 = st.radio("👥 Você prefere estudar:", ["Sozinho", "Em grupo"])
q3 = st.radio("🕓 Você é mais produtivo:", ["De manhã", "À noite"])
q4 = st.radio("📱 Você usa mais:", ["Computador", "Celular"])
q5 = st.radio("😄 Você se considera mais:", ["Introvertido", "Extrovertido"])

# --- TRADUZIR RESPOSTAS PARA NÚMEROS ---
respostas = {
    "Lendo textos": 0, "Vendo vídeos": 1,
    "Sozinho": 0, "Em grupo": 1,
    "De manhã": 0, "À noite": 1,
    "Computador": 0, "Celular": 1,
    "Introvertido": 0, "Extrovertido": 1
}

dados_usuario = pd.DataFrame([[
    respostas[q1], respostas[q2], respostas[q3],
    respostas[q4], respostas[q5]
]], columns=["Aprendizado", "Estudo", "Horario", "Dispositivo", "Personalidade"])

# --- DADOS SIMULADOS DE OUTRAS PESSOAS ---
import numpy as np
np.random.seed(42)
dados_simulados = pd.DataFrame(np.random.randint(0, 2, size=(50, 5)),
                               columns=dados_usuario.columns)

# Adiciona o usuário aos dados
dados_todos = pd.concat([dados_simulados, dados_usuario], ignore_index=True)

# --- CLUSTERING ---
kmeans = KMeans(n_clusters=3, random_state=42)
dados_todos["Grupo"] = kmeans.fit_predict(dados_todos)

# Posição do usuário
grupo_usuario = dados_todos.iloc[-1]["Grupo"]

# --- RESULTADO ---
st.success(f"✨ Você foi agrupado no **Grupo {int(grupo_usuario) + 1}**!")

# --- INTERPRETAÇÃO SIMPLES ---
descricoes = {
    0: "Pessoas mais **reflexivas**, que gostam de aprender sozinhas e preferem o computador 📚💻",
    1: "Pessoas mais **visuais e sociáveis**, que preferem vídeos e estudar em grupo 🎥👯‍♀️",
    2: "Pessoas mais **noturnas e práticas**, ligadas ao celular e ao estudo flexível 🌙📱"
}
st.info(descricoes.get(int(grupo_usuario), "Um grupo diferente e único!"))

# --- VISUALIZAÇÃO (PCA) ---
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pontos = pca.fit_transform(dados_todos.drop("Grupo", axis=1))
dados_todos["PCA1"] = pontos[:, 0]
dados_todos["PCA2"] = pontos[:, 1]

plt.figure(figsize=(8, 5))
sns.scatterplot(data=dados_todos, x="PCA1", y="PCA2", hue="Grupo", palette="Set2", s=100, alpha=0.7)

# Marca o usuário
plt.scatter(dados_todos.iloc[-1]["PCA1"], dados_todos.iloc[-1]["PCA2"],
            color='black', s=200, marker='*', label='Você')

plt.legend()
plt.title("Visualização dos Grupos Descobertos")
st.pyplot(plt)

# --- EXPLICAÇÃO FINAL ---
st.markdown("""
---

### 🧠 O que aconteceu aqui?

A estatística olhou para suas respostas e comparou com as de outras pessoas.  
Mesmo sem saber nomes, ela identificou **padrões parecidos** e criou grupos com características semelhantes.

Isso é o que **técnicas de aprendizado de máquina**, como o *K-means*, fazem todos os dias:  
agrupam dados, organizam informações, e ajudam a entender o mundo 🌍

Imagine usar isso para personalizar ensino, saúde, música, filmes... o céu é o limite 🚀
""")
