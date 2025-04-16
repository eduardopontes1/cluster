import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

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

respostas = []
for i in range(len(perguntas)):
    st.write(f"**{perguntas[i]}**")
    escolha = st.radio("", alternativas[i], key=f"q{i}")
    resposta = 0 if escolha == alternativas[i][0] else 1
    respostas.append(resposta)

if st.button("Ver resultado"):
    X_novo = np.array(respostas).reshape(1, -1)

    # Dados de referência
    grupo_humanas = np.array([
        [0,0,0,0,0], [0,0,1,0,0], [0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],
        [0,0,0,1,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],
        [0,1,1,0,0],[0,1,0,1,0],[0,1,0,0,1],[0,0,1,0,1],[0,0,0,1,1]
    ])
    grupo_exatas  = np.array([
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

    # VISUALIZAÇÃO MELHORADA
    from sklearn.decomposition import PCA

    X_vis = np.vstack((X_treino, X_novo))
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vis)

    # Centros dos clusters no espaço 2D
    centros_2d = pca.transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots()
    cores = ["blue", "red"]
    formas = ["o", "s"]

    # Previsões dos dados de treino
    labels_pred = kmeans.predict(X_treino)

    # Plotando os pontos dos grupos
    for cluster_id in range(2):
        indices = np.where(labels_pred == cluster_id)
        ax.scatter(
            X_2d[indices, 0], X_2d[indices, 1],
            c=cores[cluster_id],
            marker=formas[cluster_id],
            label=f"Grupo {rotulos[cluster_id]}",
            s=120,
            alpha=0.6,
            edgecolor='gray'
        )

    # Plotando os centros dos clusters
    ax.scatter(
        centros_2d[:, 0], centros_2d[:, 1],
        c=cores,
        marker='X',
        s=300,
        label='Centro do grupo',
        edgecolor='black'
    )

    # Plotando o ponto do usuário
    ax.scatter(
        X_2d[-1, 0], X_2d[-1, 1],
        marker=simbolo,
        color=cor,
        s=300,
        edgecolor='black',
        label="Você"
    )

    ax.set_title("Agrupamento dos perfis (Humanas x Exatas)")
    ax.axis("off")
    ax.legend()
    st.pyplot(fig)

    # Texto final
    if perfil == "Humanas":
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para comunicação, interpretação e temas sociais."
    else:
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para lógica, cálculo e pensamento analítico."

    st.subheader(f"Seu perfil é: {perfil} 🎯")
    st.info(texto)

    st.write("A técnica estatística conhecida como K-Means é amplamente utilizada em aplicativos de redes sociais como Instagram e TikTok. Já reparou que, ao criar uma conta no TikTok, ele pergunta que tipo de vídeos você gosta? Isso é parte de um processo de agrupamento, no qual o algoritmo tenta te colocar em um grupo com pessoas que têm preferências parecidas com as suas. Assim, ele identifica os estilos de vídeos que mais combinam com o seu perfil, com o objetivo de te manter engajado no aplicativo pelo maior tempo possível. Essa técnica também é usada para exibir anúncios que têm mais chance de agradar você. Entendeu agora por que às vezes aparece aquele anúncio exatamente sobre o que você estava pensando? Pois é... a estatística estava agindo o tempo todo — e você nem percebeu!")
