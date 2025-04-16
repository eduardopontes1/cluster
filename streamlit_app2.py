import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import os

st.title("Descubra seu perfil Acadêmico")
st.write("Responda às perguntas abaixo e veja em qual grupo você se encaixa com base em agrupamento!")

# Perguntas e alternativas
perguntas = [
    "Você prefere escrever uma redação ou resolver um problema de matemática?",
    "Você se interessa mais por história ou física?",
    "Você gostaria de trabalhar com pessoas ou com tecnologia?",
    "Você se sai melhor em interpretar textos ou em cálculos?",
    "Você prefere assistir a um documentário sobre política ou sobre engenharia?"
]

alternativas = [
    ("Redação", "Problema de matemática"),
    ("História", "Física"),
    ("Pessoas", "Tecnologia"),
    ("Interpretar textos", "Cálculos"),
    ("Política", "Engenharia")
]

# Coleta de respostas
respostas = []
for i in range(len(perguntas)):
    st.write(f"**{perguntas[i]}**")
    escolha = st.radio("", alternativas[i], key=f"q{i}")
    resposta = 0 if escolha == alternativas[i][0] else 1
    respostas.append(resposta)

if st.button("Ver resultado"):
    X_novo = np.array(respostas).reshape(1, -1)

    # Dados de treino (não alterar)
    grupo_humanas = np.array([
        [0,0,0,0,0], [0,0,1,0,0], [0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],
        [0,0,0,1,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],
        [0,1,1,0,0],[0,1,0,1,0],[0,1,0,0,1],[0,0,1,0,1],[0,0,0,1,1]
    ])
    grupo_exatas = np.array([
        [1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],
        [1,1,1,1,0],[0,0,1,1,1],[0,1,0,1,1],[0,1,1,0,1],[0,1,1,1,0],
        [1,0,0,1,1],[1,0,1,0,1],[1,0,1,1,0],[1,1,0,1,0],[1,1,1,0,0]
    ])
    X_treino = np.vstack((grupo_humanas, grupo_exatas))

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_treino)
    grupo = kmeans.predict(X_novo)[0]

    medias = kmeans.cluster_centers_
    rotulos = ["Humanas" if np.mean(c) < 0.5 else "Exatas" for c in medias]
    perfil = rotulos[grupo]

    cor = "blue" if perfil == "Humanas" else "red"
    simbolo = "o" if perfil == "Humanas" else "s"

    # === Armazenar os últimos 5 usuários ===
    arquivo_usuarios = "usuarios.npy"
    if os.path.exists(arquivo_usuarios):
        usuarios_previos = np.load(arquivo_usuarios)
        if len(usuarios_previos) >= 5:
            usuarios_previos = usuarios_previos[1:]  # remove o mais antigo
        usuarios_atualizados = np.vstack([usuarios_previos, X_novo])
    else:
        usuarios_atualizados = X_novo
    np.save(arquivo_usuarios, usuarios_atualizados)

    # Dados para visualização
    X_vis = np.vstack((X_treino, usuarios_atualizados))
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vis)

    fig, ax = plt.subplots()
    cores = ["blue", "red"]
    formas = ["o", "s"]

    # Plotar pontos de treino
    labels_pred = kmeans.predict(X_treino)
    for i in range(len(X_treino)):
        cluster_id = labels_pred[i]
        ax.scatter(X_2d[i, 0], X_2d[i, 1], marker=formas[cluster_id], color=cores[cluster_id], s=100, alpha=0.6)

    # Plotar usuários anteriores
    n_treino = len(X_treino)
    for j in range(len(usuarios_atualizados)):
        idx = n_treino + j
        grupo_u = kmeans.predict(usuarios_atualizados[j].reshape(1, -1))[0]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], marker=formas[grupo_u], color=cores[grupo_u], s=200, edgecolor='black')

    # Plotar centro dos clusters
    centros_2d = pca.transform(kmeans.cluster_centers_)
    for i, centro in enumerate(centros_2d):
        ax.scatter(centro[0], centro[1], marker='X', color=cores[i], s=250, edgecolor='black')

    ax.set_title("Agrupamento dos perfis (Humanas x Exatas)")
    ax.axis("on")
    st.pyplot(fig)

    # Mensagem de perfil
    if perfil == "Humanas":
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para comunicação, interpretação e temas sociais."
    else:
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para lógica, cálculo e pensamento analítico."

    st.subheader(f"Seu perfil é: {perfil} 🎯")
    st.info(texto)

    # Curiosidade sobre KMeans
    st.write("A técnica estatística conhecida como K-Means é amplamente utilizada em aplicativos de redes sociais como Instagram e TikTok. Já reparou que, ao criar uma conta no TikTok, ele pergunta que tipo de vídeos você gosta? Isso é parte de um processo de agrupamento, no qual o algoritmo tenta te colocar em um grupo com pessoas que têm preferências parecidas com as suas...")
