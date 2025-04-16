import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

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
    ("POlítica", "Engenharia")
]

# Coletando as respostas
respostas = []
for i in range(len(perguntas)):
    st.write(f"**{perguntas[i]}**")
    escolha = st.radio("", alternativas[i], key=f"q{i}")
    resposta = 0 if escolha == alternativas[i][0] else 1
    respostas.append(resposta)

# Quando clica no botão
if st.button("Ver resultado"):
    X_novo = np.array(respostas).reshape(1, -1)

    # Seus dados de treino originais
    grupo_humanas = np.array([[0,0,0,0,0], [0,0,1,0,0], [0,1,0,0,0],[1,0,0,0,0],
                              [0,0,0,0,1],[0,0,0,1,0],[1,1,0,0,0],[1,0,1,0,0],
                              [1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],[0,1,0,1,0],
                              [0,1,0,0,1],[0,0,1,0,1],[0,0,0,1,1]])
    
    grupo_exatas  = np.array([[1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1],[1,1,0,1,1],
                              [1,1,1,0,1],[1,1,1,1,0],[0,0,1,1,1],[0,1,0,1,1],
                              [0,1,1,0,1],[0,1,1,1,0],[1,0,0,1,1],[1,0,1,0,1],
                              [1,0,1,1,0],[1,1,0,1,0],[1,1,1,0,0]])

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

    # Visualização com PCA
    X_vis = np.vstack((X_treino, X_novo))
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vis)
    centroide_2d = pca.transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots()
    cores = ["blue", "red"]
    formas = ["o", "s"]

    # Plotando todos os pontos de treino com rótulo previsto
    labels_pred = kmeans.predict(X_treino)
    for i in range(len(X_treino)):
        cluster_id = labels_pred[i]
        ax.scatter(X_2d[i, 0], X_2d[i, 1],
                   color=cores[cluster_id],
                   marker=formas[cluster_id],
                   s=150,
                   alpha=0.6)

    # Ponto do usuário
    ax.scatter(X_2d[-1, 0], X_2d[-1, 1],
               marker=simbolo, color=cor,
               s=300, edgecolor='black', label="Você")

    # Centros dos clusters
    ax.scatter(centroide_2d[:, 0], centroide_2d[:, 1],
               marker='X', color=cores, s=300, edgecolor='black', label="Centros")

    ax.set_title("Agrupamento dos perfis (Humanas x Exatas)")
    ax.axis("off")
    ax.legend()
    st.pyplot(fig)

    # Resultado final
    if perfil == "Humanas":
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para comunicação, interpretação e temas sociais."
    else:
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para lógica, cálculo e pensamento analítico."

    st.subheader(f"Seu perfil é: {perfil} 🎯")
    st.info(texto)

    st.write("A técnica estatística conhecida como K-Means é amplamente utilizada em aplicativos de redes sociais como Instagram e TikTok...")
