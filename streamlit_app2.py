import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(layout="centered")
st.title("🔍 Descubra seu Perfil Acadêmico com Clustering")
st.markdown("Responda as perguntas e veja qual curso mais combina com você usando agrupamento de dados!")

# === Funções auxiliares ===
def gerar_amostras(base, n=50):
    return np.clip(np.random.normal(loc=base, scale=0.2, size=(n, len(base))), 0, 1)

# === Etapa 1: Humanas vs Exatas ===
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

referencias_fase1 = {
    "Humanas": [1, 1, 1, 1, 1],
    "Exatas":  [0, 0, 0, 0, 0],
}

X1 = np.vstack([
    gerar_amostras(referencias_fase1["Humanas"], 50),
    gerar_amostras(referencias_fase1["Exatas"], 50)
])
kmeans1 = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans1.fit(X1)
label1 = kmeans1.predict(respostas_1.reshape(1, -1))[0]

perfil = "Humanas" if label1 == 0 else "Exatas"
st.success(f"Você tem mais afinidade com a área de **{perfil}**!")

# === Etapa 2: Curso específico ===
st.subheader("Etapa 2: Qual curso combina com você?")
if perfil == "Humanas":
    perguntas = [
        "Você gostaria de ensinar em escolas ou universidades?",
        "Você se interessa por leis, justiça ou debate?",
        "Você gosta de se comunicar, gravar vídeos ou escrever publicamente?",
        "Você gostaria de cuidar da saúde mental ou física das pessoas?",
        "Você gosta de trabalhar com crianças e adolescentes?",
        "Você tem interesse em questões sociais e humanas?",
        "Você gosta de argumentar e defender ideias?",
        "Você se interessa por comunicação e marketing?",
        "Você gostaria de atuar na área da saúde?",
    ]
    cursos = ['Professor(a)', 'Direito', 'Comunicação', 'Área Médica']
    referencias = {
        "Professor(a)":     [1, 0, 0, 0, 1, 1, 0, 0, 0],
        "Direito":          [0, 1, 0, 0, 0, 1, 1, 0, 0],
        "Comunicação":      [0, 0, 1, 0, 0, 0, 1, 1, 0],
        "Área Médica":      [0, 0, 0, 1, 0, 1, 0, 0, 1],
    ]
else:
    perguntas = [
        "Você gosta de resolver problemas matemáticos?",
        "Você se interessa por computadores e tecnologia?",
        "Você tem curiosidade sobre como o universo funciona?",
        "Você prefere lidar com dados e estatísticas do que com pessoas?",
        "Você se sente confortável programando ou usando planilhas?",
        "Você gosta de entender como máquinas funcionam?",
        "Você tem interesse em pesquisa científica?",
        "Você gosta de lógica e padrões?",
        "Você gosta de automatizar tarefas ou criar sistemas?",
    ]
    cursos = ['Engenharia', 'Matemática/Estatística', 'Física', 'Computação']
    referencias = {
        "Engenharia":              [1, 0, 0, 0, 1, 1, 0, 1, 0],
        "Matemática/Estatística": [1, 0, 0, 1, 1, 0, 1, 1, 1],
        "Física":                  [1, 0, 1, 0, 0, 0, 1, 1, 0],
        "Computação":              [0, 1, 0, 1, 1, 1, 0, 1, 1],
    ]

respostas_2 = []
for pergunta in perguntas:
    r = st.radio(pergunta, ['Sim', 'Não'], key=pergunta)
    respostas_2.append(1 if r == 'Sim' else 0)
respostas_2 = np.array(respostas_2)

# Gerar amostras baseadas nas referências
X2 = np.vstack([
    gerar_amostras(v, 50) for v in referencias.values()
])
kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans2.fit(X2)
label2 = kmeans2.predict(respostas_2.reshape(1, -1))[0]
curso_final = cursos[label2]
st.info(f"Seu curso ideal é **{curso_final}**!")

# === Visualização ===
st.subheader("Visualização dos Clusters")
pca = PCA(n_components=2)
data_viz = np.vstack([X2, respostas_2])
labels_viz = np.append(kmeans2.labels_, [label2])
pca_data = pca.fit_transform(data_viz)

markers = ['s', '^', 'o', 'D']
cores = ['red', 'green', 'blue', 'orange']
fig, ax = plt.subplots(figsize=(7, 5))

for i, curso in enumerate(cursos):
    grupo = pca_data[labels_viz == i]
    ax.scatter(grupo[:-1, 0], grupo[:-1, 1], marker=markers[i], color=cores[i], label=curso, alpha=0.6, edgecolor='k')

# Destacar o usuário
ax.scatter(pca_data[-1, 0], pca_data[-1, 1], color='black', s=120, label='Você', edgecolor='yellow', linewidth=2)
ax.set_title("Clusters de Cursos com KMeans")
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.legend()
st.pyplot(fig)

st.caption("O gráfico mostra como suas respostas se agrupam com perfis típicos de cada curso.")
