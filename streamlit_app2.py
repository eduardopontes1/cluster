os dados foram pegues do banco de dados da pnad pesquisa nacional por amostra de domicilios do ibge no site:

https://console.cloud.google.com/bigquery?p=basedosdados&d=br_ibge_pnadc&t=rendimentos_outras_fontes&page=table&invt=Abt4-A&project=dados-pnade&supportedpurview=project&ws=!1m10!1m4!4m3!1sbasedosdados!2sbr_ibge_pnadc!3srendimentos_outras_fontes!1m4!1m3!1sdados-pnade!2sbquxjob_75c6cb4a_19602e84a58!3sUS

os dados originais possuiam 1429834 de linhas, porém foram usados apenas dados referentes ao estado do ceará



import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
st.title("Descubra seu perfil Acadêmico")
st.write("Responda às perguntas abaixo e veja em qual grupo você se encaixa com base em agrupmento!")
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
    grupo_humanas = np.array([[0,0,0,0,0], [0,0,1,0,0], [0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],[0,1,0,1,0],[0,1,0,0,1],[0,0,1,0,1],[0,0,0,1,1]])
    grupo_exatas  = np.array([[1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0],[0,0,1,1,1],[0,1,0,1,1],[0,1,1,0,1],[0,1,1,1,0],[1,0,0,1,1],[1,0,1,0,1],[1,0,1,1,0],[1,1,0,1,0],[1,1,1,0,0]])
    X_treino = np.vstack((grupo_humanas, grupo_exatas))
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_treino)
    grupo = kmeans.predict(X_novo)[0]
    medias = kmeans.cluster_centers_
    rotulos = ["Humanas" if np.mean(c) < 0.5 else "Exatas" for c in medias]
    perfil = rotulos[grupo]
    cor = "blue" if perfil == "Humanas" else "red"
    simbolo = "o" if perfil == "Humanas" else "s"
    from sklearn.decomposition import PCA
    X_vis = np.vstack((X_treino, X_novo))
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vis)
    fig, ax = plt.subplots()
    cores = ["blue", "red"]
    formas = ["o", "s"]
    pontos_humanas = np.array([[0,0,0,0,0], [0,0,1,0,0], [0,1,0,0,0],[1,0,0,0,0]])
    pontos_exatas = np.array([[1,1,1,1,1], [0,1,1,1,1], [1,0,1,1,1],[1,1,0,1,1]])
    X_visual = np.vstack((pontos_humanas, pontos_exatas))
        
    labels_pred = kmeans.predict(X_visual)
    for i in range(len(X_visual)):
        cluster_id = labels_pred[i]
        ax.scatter(X_2d[i, 0], X_2d[i, 1], marker=formas[cluster_id], color=cores[cluster_id], s=100, alpha=0.6)
    #ponto daresposta
    ax.scatter(X_2d[-1, 0], X_2d[-1, 1], marker=simbolo, color=cor, s=300, edgecolor='black', label="Você")
    ax.set_title("Agrupamento dos perfis (Humanas x Exatas)")
    ax.axis("off")
    st.pyplot(fig)
    if perfil == "Humanas":
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para comunicação, interpretação e temas sociais."
    else:
        texto = "Você foi agrupado com outros alunos com perfil mais voltado para lógica, cálculo e pensamento analítico."

    st.subheader(f"Seu perfil é: {perfil} 🎯")
    st.info(texto)
    st.write("A técnica estatística conhecida como K-Means é amplamente utilizada em aplicativos de redes sociais como Instagram e TikTok. Já reparou que, ao criar uma conta no TikTok, ele pergunta que tipo de vídeos você gosta? Isso é parte de um processo de agrupamento, no qual o algoritmo tenta te colocar em um grupo com pessoas que têm preferências parecidas com as suas. Assim, ele identifica os estilos de vídeos que mais combinam com o seu perfil, com o objetivo de te manter engajado no aplicativo pelo maior tempo possível. Essa técnica também é usada para exibir anúncios que têm mais chance de agradar você. Entendeu agora por que às vezes aparece aquele anúncio exatamente sobre o que você estava pensando? Pois é... a estatística estava agindo o tempo todo — e você nem percebeu!")


