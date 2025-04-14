import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="centered")
st.title("🎓 Descubra seu Perfil Acadêmico")
st.markdown("Responda às perguntas para saber qual curso combina mais com você!")

# Funções auxiliares
def load_data(filename, shape, simulate=False):
    if os.path.exists(filename):
        return pd.read_csv(filename).values
    elif simulate:
        return np.random.randint(0, 2, size=(20, shape))  # simula 20 respostas binárias
    else:
        return np.empty((0, shape))

def save_response(filename, new_data):
    df = pd.DataFrame(new_data.reshape(1, -1))
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# Função para aplicar pesos às respostas
def aplicar_pesos(respostas, pesos):
    return np.dot(respostas, pesos)

# --- Etapa 1: Humanas ou Exatas ---
st.subheader("Etapa 1: Seu estilo de pensamento")

perguntas1 = [
    "Você prefere escrever ou resolver problemas lógicos?",
    "Você se interessa mais por leitura ou matemática?",
    "Você prefere discutir ideias ou construir coisas?",
    "Você gosta mais de interpretar textos ou fazer cálculos?",
    "Você prefere trabalhar com pessoas ou com números?"
]
opcoes1 = [
    ['Escrever', 'Resolver problemas'],
    ['Leitura', 'Matemática'],
    ['Discutir ideias', 'Construir coisas'],
    ['Interpretar textos', 'Fazer cálculos'],
    ['Pessoas', 'Números']
]
pesos1 = np.array([1.2, 1.0, 1.1, 1.0, 1.3])  # pesos para perguntas

respostas_raw1 = []
for i, p in enumerate(perguntas1):
    resp = st.radio(p, opcoes1[i], key=f"p1_{i}")
    respostas_raw1.append(1 if resp == opcoes1[i][0] else 0)

respostas1 = np.array(respostas_raw1)
score = aplicar_pesos(respostas1, pesos1)

# Teste de hipótese simples (score > limiar → Humanas)
limiar_score = np.mean(pesos1)
perfil = "Humanas" if score >= limiar_score else "Exatas"
st.success(f"Você tem mais afinidade com a área de **{perfil}**!")

# Aprendizado incremental da Etapa 1
X1 = load_data("respostas_fase1.csv", 5, simulate=True)
X1 = np.vstack([X1, respostas1])
save_response("respostas_fase1.csv", respostas1)

# Clustering da etapa 1 (apenas para visualização)
kmeans1 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels1 = kmeans1.fit_predict(X1)

# --- Etapa 2: Curso Específico ---
st.subheader("Etapa 2: Qual curso combina com você?")

# Perguntas, cursos e pesos por área
perguntas_humanas = [
    "Você gostaria de ensinar em escolas ou universidades?",
    "Você se interessa por leis, justiça ou debate?",
    "Você gosta de se comunicar, gravar vídeos ou escrever publicamente?",
    "Você gostaria de cuidar da saúde das pessoas?"
]
perguntas_exatas = [
    "Você gosta de resolver problemas matemáticos?",
    "Você se interessa por computadores e tecnologia?",
    "Você tem curiosidade sobre como o universo funciona?",
    "Você prefere lidar com dados e estatísticas do que com pessoas?"
]
cursos_humanas = ['Professor(a)', 'Direito', 'Comunicação', 'Área Médica']
cursos_exatas = ['Estatística/Matemática', 'Engenharia', 'Física', 'Ciência da Computação']
pesos2 = np.array([1.1, 1.2, 1.0, 1.3])  # mesmos para ambos por simplicidade

respostas_raw2 = []
if perfil == "Humanas":
    for i, p in enumerate(perguntas_humanas):
        resp = st.radio(p, ['Sim', 'Não'], key=f"h{i}")
        respostas_raw2.append(1 if resp == 'Sim' else 0)
    cursos = cursos_humanas
    filename2 = "respostas_humanas.csv"
else:
    for i, p in enumerate(perguntas_exatas):
        resp = st.radio(p, ['Sim', 'Não'], key=f"e{i}")
        respostas_raw2.append(1 if resp == 'Sim' else 0)
    cursos = cursos_exatas
    filename2 = "respostas_exatas.csv"

respostas2 = np.array(respostas_raw2)
X2 = load_data(filename2, 4, simulate=True)
X2 = np.vstack([X2, respostas2])
save_response(filename2, respostas2)

kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels2 = kmeans2.fit_predict(X2)
curso_idx = labels2[-1]
curso_final = cursos[curso_idx]
st.info(f"Você tem mais perfil para o curso de **{curso_final}**!")

# --- Visualização ---
st.subheader("Visualização dos Agrupamentos")
pca1 = PCA(n_components=2)
pca_data1 = pca1.fit_transform(X1)
pca2 = PCA(n_components=2)
pca_data2 = pca2.fit_transform(X2)

shapes = ['o', 's', '^', 'D', 'P', '*']  # círculos, quadrados, triângulos, etc.
cores = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Etapa 1
for i in np.unique(labels1):
    ax1.scatter(
        pca_data1[labels1 == i, 0],
        pca_data1[labels1 == i, 1],
        label=f'Grupo {i+1}',
        alpha=0.6,
        edgecolor='k',
        marker=shapes[i % len(shapes)],
        c=cores[i % len(cores)]
    )
ax1.set_title("Fase 1: Humanas vs Exatas")
ax1.legend()

# Etapa 2
for i, curso in enumerate(cursos):
    ax2.scatter(
        pca_data2[labels2 == i, 0],
        pca_data2[labels2 == i, 1],
        label=curso,
        alpha=0.6,
        edgecolor='k',
        marker=shapes[i % len(shapes)],
        c=cores[i % len(cores)]
    )
ax2.set_title("Fase 2: Cursos dentro da área")
ax2.legend()

st.pyplot(fig)

st.caption("Os gráficos mostram os agrupamentos formados com base nas suas respostas. Quanto mais o sistema for usado, melhor ele entenderá os perfis!")
