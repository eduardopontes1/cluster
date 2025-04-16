import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

# Configuração da página
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico")
st.write("Responda às perguntas abaixo e descubra em qual grupo você se encaixa usando Machine Learning!")

# Dados das perguntas (organizados em dicionário para facilitar manutenção)
perguntas = {
    "Pergunta 1": ("Redação", "Problema de matemática"),
    "Pergunta 2": ("História", "Física"),
    "Pergunta 3": ("Pessoas", "Tecnologia"),
    "Pergunta 4": ("Interpretar textos", "Cálculos"),
    "Pergunta 5": ("Política", "Engenharia")
}

# Coleta de respostas
respostas = []
for pergunta, (opcao1, opcao2) in perguntas.items():
    st.write(f"**{pergunta}**")
    escolha = st.radio("", [opcao1, opcao2], key=pergunta)
    respostas.append(0 if escolha == opcao1 else 1)

if st.button("🔎 Ver resultado"):
    X_novo = np.array(respostas).reshape(1, -1)

    # Dados de treino simulados (com mais variação)
    grupo_humanas = np.array([
        [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 1, 0]  # Adicionei alunos "mistos"
    ])
    grupo_exatas = np.array([
        [1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1]
    ])
    X_treino = np.vstack((grupo_humanas, grupo_exatas))
    y_treino = np.array([0] * len(grupo_humanas) + [1] * len(grupo_exatas))  # 0=Humanas, 1=Exatas

    # K-means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_treino)

    # Mapeamento dos clusters para os rótulos reais (baseado nos dados de treino)
    rotulos = ["Humanas", "Exatas"]
    cluster_para_rotulo = {
        0: rotulos[y_treino[kmeans.labels_ == 0][0]],  # Pega o rótulo majoritário no cluster 0
        1: rotulos[y_treino[kmeans.labels_ == 1][0]]   # Pega o rótulo majoritário no cluster 1
    }
    perfil = cluster_para_rotulo[kmeans.predict(X_novo)[0]]

    # Silhouette Score (qualidade do agrupamento)
    score = silhouette_score(X_treino, kmeans.labels_)
    
    # Visualização com PCA
    X_vis = np.vstack((X_treino, X_novo))
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vis)

    # Cores e marcadores
    cores = ["blue", "red"]
    marcadores = ["o", "s"]  # Círculo para Humanas, Quadrado para Exatas

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot dos dados de treino
    for i in range(len(X_treino)):
        cluster = kmeans.labels_[i]
        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            marker=marcadores[cluster],
            color=cores[cluster],
            alpha=0.6,
            label=f"{rotulos[y_treino[i]]} (Treino)" if i == 0 else ""
        )

    # Plot do novo aluno (destaque)
    ax.scatter(
        X_2d[-1, 0], X_2d[-1, 1],
        marker="*",  # Estrela para o usuário
        color="green",
        s=200,
        edgecolor="black",
        label="Você"
    )

    # Configurações do gráfico
    ax.set_title("Agrupamento de Perfis Acadêmicos (K-means + PCA)", pad=20)
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    # Exibição dos resultados
    st.pyplot(fig)
    st.success(f"**Seu perfil é:** {perfil} 🎯")
    
    if perfil == "Humanas":
        st.info("""
        Você foi agrupado com alunos que preferem **comunicação, interpretação e temas sociais**.  
        Características típicas:  
        - Habilidade em escrita e análise crítica  
        - Interesse em humanidades e artes  
        """)
    else:
        st.info("""
        Você foi agrupado com alunos que preferem **lógica, cálculo e pensamento analítico**.  
        Características típicas:  
        - Raciocínio quantitativo  
        - Aptidão para ciências exatas e tecnologia  
        """)
    
    st.metric("Qualidade do Agrupamento (Silhouette Score)", f"{score:.2f}",
              help="Valores próximos de 1 indicam clusters bem definidos")

    # Explicação técnica (opcional)
    with st.expander("ℹ️ Como funciona?"):
        st.markdown("""
        - **K-means**: Algoritmo de Machine Learning que agrupa dados similares.  
        - **PCA**: Reduz a dimensionalidade para visualização em 2D.  
        - **Silhouette Score**: Mede quão bem cada ponto se encaixa no seu cluster.  
        """)
