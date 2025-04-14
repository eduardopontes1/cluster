import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("🔍 Descubra seu Perfil Acadêmico")
st.markdown("Responda algumas perguntas e veja em qual grupo você se encaixa!")

# ---------------------------- Funções auxiliares ----------------------------
def aplicar_kmeans(dados, n_clusters=2):
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = modelo.fit_predict(dados)
    return labels, modelo

def projetar_pca(dados):
    pca = PCA(n_components=2)
    return pca.fit_transform(dados)

def formatar_resposta(respostas, pesos):
    return np.array([respostas[i] * pesos[i] for i in range(len(respostas))])

# ---------------------------- Etapa 1 ----------------------------
st.subheader("Etapa 1: Seu estilo de pensamento")

# Perguntas e pesos
perguntas_1 = [
    "Você prefere escrever ou resolver problemas lógicos?",
    "Você se interessa mais por leitura ou matemática?",
    "Você prefere discutir ideias ou construir coisas?",
    "Você gosta mais de interpretar textos ou fazer cálculos?",
    "Você prefere trabalhar com pessoas ou com números?"
]

opcoes_1 = [
    ["Escrever", "Resolver problemas"],
    ["Leitura", "Matemática"],
    ["Discutir ideias", "Construir coisas"],
    ["Interpretar textos", "Fazer cálculos"],
    ["Pessoas", "Números"]
]

pesos_1 = [1, 1, 1, 1, 1]  # todos os pesos iguais por simplicidade
respostas_brutas_1 = []

for i, pergunta in enumerate(perguntas_1):
    resposta = st.radio(pergunta, opcoes_1[i], key=f"etapa1_q{i}")
    respostas_brutas_1.append(1 if resposta in ["Escrever", "Leitura", "Discutir ideias", "Interpretar textos", "Pessoas"] else 0)

resposta_formatada_1 = formatar_resposta(respostas_brutas_1, pesos_1)

# Grupos fixos simulados para Humanas e Exatas
referencias_etapa1 = {
    'Humanas': [1, 1, 1, 1, 1],
    'Exatas': [0, 0, 0, 0, 0]
}

data_etapa1 = np.array(list(referencias_etapa1.values()) + [resposta_formatada_1])
labels_1, modelo_1 = aplicar_kmeans(data_etapa1, n_clusters=2)
perfil_usuario = labels_1[-1]
perfil = list(referencias_etapa1.keys())[perfil_usuario]

st.success(f"Você tem mais afinidade com a área de **{perfil}**!")

# ---------------------------- Etapa 2 ----------------------------
st.subheader("Etapa 2: Qual curso combina com você?")

if perfil == "Humanas":
    perguntas_2 = [
        "Você gostaria de ensinar em escolas ou universidades?",
        "Você se interessa por leis, justiça ou debate?",
        "Você gosta de se comunicar, gravar vídeos ou escrever publicamente?",
        "Você gostaria de cuidar da saúde das pessoas?",
        "Você gosta de ler e interpretar textos complexos?",
        "Você sente empatia e deseja ajudar pessoas emocionalmente?",
        "Você prefere ambientes sociais a ambientes técnicos?",
        "Você se interessa por filosofia ou sociologia?",
        "Você se imagina trabalhando com jornalismo ou mídias sociais?"
    ]

    cursos_humanas = {
        'Professor(a)': [1, 0, 0, 0, 1, 1, 0, 0, 0],
        'Direito': [1, 1, 0, 0, 0, 0, 0, 1, 0],
        'Comunicação': [1, 0, 1, 0, 0, 0, 1, 0, 1],
        'Área Médica': [0, 0, 0, 1, 0, 1, 0, 0, 0]
    }
    opcoes_binarias = ["Sim", "Não"]
    respostas_2 = []
    for i, pergunta in enumerate(perguntas_2):
        r = st.radio(pergunta, opcoes_binarias, key=f"humanas_q{i}")
        respostas_2.append(1 if r == "Sim" else 0)

    referencia = np.array(list(cursos_humanas.values()))
    dados = np.vstack([referencia, respostas_2])
    labels_2, modelo_2 = aplicar_kmeans(dados, n_clusters=4)
    grupo_usuario = labels_2[-1]
    curso_final = list(cursos_humanas.keys())[grupo_usuario % len(cursos_humanas)]
    st.info(f"Você tem mais perfil para o curso de **{curso_final}**!")

    # Visualização
    proj = projetar_pca(dados)
    cores = ['red', 'blue', 'green', 'purple']
    formas = ['s', '^', 'D', 'o']
    fig, ax = plt.subplots()
    for i, curso in enumerate(cursos_humanas):
        ax.scatter(proj[i, 0], proj[i, 1], color=cores[i], marker=formas[i], label=curso, s=100)
    ax.scatter(proj[-1, 0], proj[-1, 1], color='black', marker='x', s=150, label='Você')
    ax.set_title("Agrupamento dos cursos de Humanas")
    ax.legend()
    st.pyplot(fig)

else:
    perguntas_2 = [
        "Você gosta de resolver problemas matemáticos?",
        "Você se interessa por computadores e tecnologia?",
        "Você tem curiosidade sobre como o universo funciona?",
        "Você prefere lidar com dados e estatísticas do que com pessoas?",
        "Você gosta de lógica e raciocínio abstrato?",
        "Você prefere ambientes estruturados a caóticos?",
        "Você se interessa por engenharia e construção?",
        "Você gosta de desafios intelectuais complexos?",
        "Você se imagina criando softwares ou sistemas?"
    ]

    cursos_exatas = {
        'Estatística/Matemática': [1, 0, 0, 1, 1, 1, 0, 1, 0],
        'Engenharia': [1, 1, 0, 0, 1, 1, 1, 0, 0],
        'Física': [1, 0, 1, 0, 1, 1, 0, 1, 0],
        'Ciência da Computação': [0, 1, 0, 1, 1, 1, 0, 1, 1]
    }

    opcoes_binarias = ["Sim", "Não"]
    respostas_2 = []
    for i, pergunta in enumerate(perguntas_2):
        r = st.radio(pergunta, opcoes_binarias, key=f"exatas_q{i}")
        respostas_2.append(1 if r == "Sim" else 0)

    referencia = np.array(list(cursos_exatas.values()))
    dados = np.vstack([referencia, respostas_2])
    labels_2, modelo_2 = aplicar_kmeans(dados, n_clusters=4)
    grupo_usuario = labels_2[-1]
    curso_final = list(cursos_exatas.keys())[grupo_usuario % len(cursos_exatas)]
    st.info(f"Você tem mais perfil para o curso de **{curso_final}**!")

    # Visualização
    proj = projetar_pca(dados)
    cores = ['red', 'blue', 'green', 'purple']
    formas = ['s', '^', 'D', 'o']
    fig, ax = plt.subplots()
    for i, curso in enumerate(cursos_exatas):
        ax.scatter(proj[i, 0], proj[i, 1], color=cores[i], marker=formas[i], label=curso, s=100)
    ax.scatter(proj[-1, 0], proj[-1, 1], color='black', marker='x', s=150, label='Você')
    ax.set_title("Agrupamento dos cursos de Exatas")
    ax.legend()
    st.pyplot(fig)

st.caption("\n\nCada ponto representa um perfil de curso. O agrupamento é feito com base nas respostas e referenciais típicos de cada área.")
