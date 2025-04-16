import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Configuração da página
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico")
st.write("Marque os conteúdos com que você mais se identifica:")

# Itens para seleção (opções reformuladas)
itens = [
    {"texto": "Escrever poemas ou crônicas", "valor": 0},
    {"texto": "Resolver desafios de programação", "valor": 1},
    {"texto": "Debater sobre filosofia ou sociologia", "valor": 0},
    {"texto": "Projetar experimentos científicos", "valor": 1},
    {"texto": "Analisar obras de arte", "valor": 0},
    {"texto": "Desenvolver fórmulas matemáticas", "valor": 1},
    {"texto": "Ler sobre política internacional", "valor": 0},
    {"texto": "Estudar novas tecnologias", "valor": 1}
]

# Coleta de respostas (checkbox para múltipla escolha)
respostas = [0] * len(itens)
for i, item in enumerate(itens):
    if st.checkbox(item["texto"], key=f"item_{i}"):
        respostas[i] = 1  # Marca como "identificado"

if st.button("🔎 Descobrir meu perfil"):
    X_novo = np.array(respostas).reshape(1, -1)

    # Dados de treino simulados (Humanas = 0, Exatas = 1)
    grupo_humanas = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0],  # Perfil totalmente humanas
        [1, 0, 1, 0, 0, 0, 1, 0],   # Perfil humanas com menos arte
        [0, 0, 1, 0, 1, 0, 1, 0]    # Menos literatura, mais humanidades
    ])
    grupo_exatas = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],  # Perfil totalmente exatas
        [0, 1, 0, 1, 0, 0, 0, 1],  # Menos matemática, mais tecnologia
        [0, 0, 0, 1, 0, 1, 0, 1]   # Foco em ciências/tecnologia
    ])
    X_treino = np.vstack((grupo_humanas, grupo_exatas))
    y_treino = np.array([0] * len(grupo_humanas) + [1] * len(grupo_exatas))  # Parêntese fechado aqui

    # K-means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_treino)

    # Mapeamento dos clusters
    perfil = "Humanas" if kmeans.predict(X_novo)[0] == 0 else "Exatas"

    # Visualização com PCA
    X_vis = np.vstack((X_treino, X_novo))
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vis)

    # Configuração do gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Cores e marcadores
    cores = ["blue", "red"]
    marcadores = ["o", "s"]  # Círculo (Humanas), Quadrado (Exatas)

    # Plot dos dados de treino
    for i in range(len(X_treino)):
        cluster = kmeans.labels_[i]
        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            marker=marcadores[cluster],
            color=cores[cluster],
            alpha=0.6,
            label=f"{'Humanas' if y_treino[i] == 0 else 'Exatas'} (Referência)" if i == 0 else ""
        )

    # Plot do usuário (destaque)
    ax.scatter(
        X_2d[-1, 0], X_2d[-1, 1],
        marker="*",  # Estrela
        color="green",
        s=200,
        edgecolor="black",
        label="Você"
    )

    # Ajustes estéticos
    ax.set_title("Mapa de Perfis Acadêmicos", pad=20)
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    # Exibição dos resultados
    st.pyplot(fig)
    st.success(f"**Seu perfil predominante é:** {perfil} 🎯")
    
    if perfil == "Humanas":
        st.info("""
        **Características do seu perfil:**  
        ✍️ Habilidade em escrita e comunicação  
        🌍 Interesse em ciências humanas e artes  
        🧠 Pensamento crítico e analógico  
        """)
    else:
        st.info("""
        **Características do seu perfil:**  
        ➗ Aptidão para lógica e matemática  
        🔬 Interesse em ciências e tecnologia  
        ⚙️ Raciocínio quantitativo e analítico  
        """)

    # Explicação simplificada
    with st.expander("ℹ️ Como isso funciona?"):
        st.markdown("""
        - **Seleção**: Você marcou os conteúdos que mais gosta.  
        - **Agrupamento**: O algoritmo K-means comparou suas respostas com perfis de referência.  
        - **Resultado**: A posição no gráfico mostra seu grupo mais próximo.  
        """) 
