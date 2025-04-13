import streamlit as st 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="centered")
st.title("🔍 Descubra seu Perfil Acadêmico")
st.markdown("Responda algumas perguntas e veja em qual grupo você se encaixa!")

# --- Função para carregar dados anteriores (para aprendizado incremental simples) ---
def load_data(filename, shape):
    if os.path.exists(filename):
        return pd.read_csv(filename).values
    else:
        return np.empty((0, shape))

# --- Função para salvar nova resposta ---
def save_response(filename, new_data):
    df = pd.DataFrame(new_data.reshape(1, -1))
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# --- Parte 1: Perguntas para identificar Humanas ou Exatas ---
st.subheader("Etapa 1: Seu estilo de pensamento")
p1 = st.radio("Você prefere escrever ou resolver problemas lógicos?", ['Escrever', 'Resolver problemas'])
p2 = st.radio("Você se interessa mais por leitura ou matemática?", ['Leitura', 'Matemática'])
p3 = st.radio("Você prefere discutir ideias ou construir coisas?", ['Discutir ideias', 'Construir coisas'])
p4 = st.radio("Você gosta mais de interpretar textos ou fazer cálculos?", ['Interpretar textos', 'Fazer cálculos'])
p5 = st.radio("Você prefere trabalhar com pessoas ou com números?", ['Pessoas', 'Números'])

respostas_1 = np.array([
    1 if p1 == 'Escrever' else 0,
    1 if p2 == 'Leitura' else 0,
    1 if p3 == 'Discutir ideias' else 0,
    1 if p4 == 'Interpretar textos' else 0,
    1 if p5 == 'Pessoas' else 0
])

X1 = load_data("respostas_fase1.csv", 5)
X1 = np.vstack([X1, respostas_1])
save_response("respostas_fase1.csv", respostas_1)

kmeans1 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels1 = kmeans1.fit_predict(X1)
user_group = labels1[-1]

perfil = "Humanas" if user_group == 0 else "Exatas"
st.success(f"Você tem mais afinidade com a área de **{perfil}**!")

# --- Parte 2: Perguntas específicas por área ---
st.subheader("Etapa 2: Qual curso combina com você?")

if perfil == "Humanas":
    h1 = st.radio("Você gostaria de ensinar em escolas ou universidades?", ['Sim', 'Não'])
    h2 = st.radio("Você se interessa por leis, justiça ou debate?", ['Sim', 'Não'])
    h3 = st.radio("Você gosta de se comunicar, gravar vídeos ou escrever publicamente?", ['Sim', 'Não'])
    h4 = st.radio("Você gostaria de cuidar da saúde das pessoas?", ['Sim', 'Não'])

    respostas_2 = np.array([
        1 if h1 == 'Sim' else 0,
        1 if h2 == 'Sim' else 0,
        1 if h3 == 'Sim' else 0,
        1 if h4 == 'Sim' else 0
    ])

    X2 = load_data("respostas_humanas.csv", 4)
    X2 = np.vstack([X2, respostas_2])
    save_response("respostas_humanas.csv", respostas_2)

    kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels2 = kmeans2.fit_predict(X2)
    curso_idx = labels2[-1]
    cursos = ['Professor(a)', 'Direito', 'Comunicação', 'Área Médica']
    curso_final = cursos[curso_idx]
    st.info(f"Você tem mais perfil para o curso de **{curso_final}**!")

else:
    e1 = st.radio("Você gosta de resolver problemas matemáticos?", ['Sim', 'Não'])
    e2 = st.radio("Você se interessa por computadores e tecnologia?", ['Sim', 'Não'])
    e3 = st.radio("Você tem curiosidade sobre como o universo funciona?", ['Sim', 'Não'])
    e4 = st.radio("Você prefere lidar com dados e estatísticas do que com pessoas?", ['Sim', 'Não'])

    respostas_2 = np.array([
        1 if e1 == 'Sim' else 0,
        1 if e2 == 'Sim' else 0,
        1 if e3 == 'Sim' else 0,
        1 if e4 == 'Sim' else 0
    ])

    X2 = load_data("respostas_exatas.csv", 4)
    X2 = np.vstack([X2, respostas_2])
    save_response("respostas_exatas.csv", respostas_2)

    kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels2 = kmeans2.fit_predict(X2)
    curso_idx = labels2[-1]
    cursos = ['Estatística/Matemática', 'Engenharia', 'Física', 'Ciência da Computação']
    curso_final = cursos[curso_idx]
    st.info(f"Você tem mais perfil para o curso de **{curso_final}**!")

# --- Visualização ---
st.subheader("Visualização dos Agrupamentos")
pca1 = PCA(n_components=2)
pca_data1 = pca1.fit_transform(X1)
pca2 = PCA(n_components=2)
pca_data2 = pca2.fit_transform(X2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
scatter1 = ax1.scatter(pca_data1[:, 0], pca_data1[:, 1], c=labels1, cmap='coolwarm', alpha=0.6, edgecolor='k', marker='o')
ax1.set_title("Fase 1: Humanas vs Exatas")
ax1.set_xlabel("Eixo X: Preferências cognitivas")
ax1.set_ylabel("Eixo Y: Estilo de raciocínio")

scatter2 = ax2.scatter(pca_data2[:, 0], pca_data2[:, 1], c=labels2, cmap='tab10', alpha=0.6, edgecolor='k', marker='o')
ax2.set_title("Fase 2: Perfil dentro da área escolhida")
ax2.set_xlabel("Eixo X: Interesse específico")
ax2.set_ylabel("Eixo Y: Afinidade comportamental")

st.pyplot(fig)

st.caption("\n\nCada ponto representa uma pessoa. Os agrupamentos são formados com base nas respostas dadas. Com o tempo, o sistema aprende com os usuários!")
