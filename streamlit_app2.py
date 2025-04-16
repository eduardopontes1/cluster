# Após suas importações
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Inicializar fila de usuários se ainda não existir
if "fila_usuarios" not in st.session_state:
    st.session_state.fila_usuarios = []

# Após o botão "Ver resultado":
if st.button("Ver resultado"):
    X_novo = np.array(respostas).reshape(1, -1)

    # Dados de treino
    grupo_humanas = np.array([
        [0,0,0,0,0], [0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0], [0,0,0,0,1],
        [0,0,0,1,0], [1,1,0,0,0], [1,0,1,0,0], [1,0,0,1,0], [1,0,0,0,1],
        [0,1,1,0,0], [0,1,0,1,0], [0,1,0,0,1], [0,0,1,0,1], [0,0,0,1,1]
    ])
    grupo_exatas = np.array([
        [1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1], [1,1,0,1,1], [1,1,1,0,1],
        [1,1,1,1,0], [0,0,1,1,1], [0,1,0,1,1], [0,1,1,0,1], [0,1,1,1,0],
        [1,0,0,1,1], [1,0,1,0,1], [1,0,1,1,0], [1,1,0,1,0], [1,1,1,0,0]
    ])

    X_treino = np.vstack((grupo_humanas, grupo_exatas))

    # Atualiza a fila de usuários
    st.session_state.fila_usuarios.append(X_novo[0])
    if len(st.session_state.fila_usuarios) > 5:
        st.session_state.fila_usuarios.pop(0)

    # Adiciona os pontos anteriores dos usuários à visualização, mas não ao treino
    X_plot = np.vstack((X_treino, np.array(st.session_state.fila_usuarios), X_novo))

    # Treinamento com dados originais (sem incluir usuários anteriores)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_treino)

    # Previsão do novo usuário
    grupo = kmeans.predict(X_novo)[0]

    # Definindo o rótulo
    medias = kmeans.cluster_centers_
    rotulos = ["Humanas" if np.mean(c) < 0.5 else "Exatas" for c in medias]
    perfil = rotulos[grupo]

    cor = "blue" if perfil == "Humanas" else "red"
    simbolo = "o" if perfil == "Humanas" else "s"

    # Redução de dimensionalidade
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_plot)

    fig, ax = plt.subplots()
    cores = ["blue", "red"]
    formas = ["o", "s"]

    # Labels para dados do plot (treino + usuários anteriores)
    labels_pred = kmeans.predict(X_plot[:-1])  # exclui o novo

    for i in range(len(X_plot)-1):  # Exclui o último que é o novo usuário
        cluster_id = labels_pred[i]
        ax.scatter(X_2d[i, 0], X_2d[i, 1], marker=formas[cluster_id], color=cores[cluster_id], s=100, alpha=0.6)

    # Ponto do novo usuário
    ax.scatter(X_2d[-1, 0], X_2d[-1, 1], marker=simbolo, color=cor, s=300, edgecolor='black', label=\"Você\")

    # Centros dos clusters
    centros_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centros_2d[:, 0], centros_2d[:, 1], c='black', marker='X', s=200, label='Centro')

    ax.set_title(\"Agrupamento dos perfis (Humanas x Exatas)\")
    ax.axis(\"on\")
    st.pyplot(fig)

    st.subheader(f\"Seu perfil é: {perfil} 🎯\")
    if perfil == \"Humanas\":
        st.info(\"Você foi agrupado com outros alunos com perfil mais voltado para comunicação, interpretação e temas sociais.\")
    else:
        st.info(\"Você foi agrupado com outros alunos com perfil mais voltado para lógica, cálculo e pensamento analítico.\")
