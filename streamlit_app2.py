import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Configuração da página
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico")
st.write("Marque os conteúdos com que você mais se identifica:")

# --- PRIMEIRA ETAPA: CLASSIFICAÇÃO INICIAL ---
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

respostas = [0] * len(itens)
for i, item in enumerate(itens):
    if st.checkbox(item["texto"], key=f"item_{i}"):
        respostas[i] = 1

if st.button("🔎 Descobrir meu perfil"):
    if sum(respostas) == 0:
        st.warning("Por favor, selecione pelo menos um conteúdo.")
    else:
        X_novo = np.array(respostas).reshape(1, -1)
        
        # Dados de treino
        grupo_humanas = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0]
        ])
        grupo_exatas = np.array([
            [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0]
        ])
        X_treino = np.vstack((grupo_humanas, grupo_exatas))
        y_treino = np.array([0] * len(grupo_humanas) + [1] * len(grupo_exatas))
        
        # K-means
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(X_treino)
        perfil = "Humanas" if kmeans.predict(X_novo)[0] == 0 else "Exatas"
        
        # Visualização
        X_vis = np.vstack((X_treino, X_novo))
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_vis)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cores = ["blue", "red"]
        marcadores = ["o", "s"]
        
        for i in range(len(X_treino)):
            cluster = kmeans.labels_[i]
            ax.scatter(
                X_2d[i, 0], X_2d[i, 1],
                marker=marcadores[cluster],
                color=cores[cluster],
                alpha=0.6,
                label=f"{'Humanas' if y_treino[i] == 0 else 'Exatas'}" if i == 0 else ""
            )
        
        ax.scatter(X_2d[-1, 0], X_2d[-1, 1], marker="*", color="green", s=200, label="Você")
        ax.set_title("Mapa de Perfis Acadêmicos", pad=20)
        ax.legend()
        st.pyplot(fig)
        
        st.success(f"**Seu perfil predominante é:** {perfil} 🎯")
        if perfil == "Humanas":
            st.info("""
            **Características:**  
            ✍️ Habilidade em escrita e comunicação  
            🌍 Interesse em humanidades e artes  
            🧠 Pensamento crítico e analógico  
            """)
        else:
            st.info("""
            **Características:**  
            ➗ Aptidão para lógica e matemática  
            🔬 Interesse em ciências e tecnologia  
            ⚙️ Raciocínio quantitativo  
            """)

        # --- SEGUNDA ETAPA: SUGESTÃO DE CURSOS COM ÍCONES ---
        st.divider()
        st.subheader("📚 Cursos Recomendados")
        
        if perfil == "Humanas":
            cursos = [
                ("Letras", "📖"), ("História", "🏛️"), ("Direito", "⚖️"), 
                ("Psicologia", "🧠"), ("Artes Visuais", "🎨")
            ]
            st.write("Selecione até **3 cursos** de Humanas que mais te interessam:")
        else:
            cursos = [
                ("Estatística", "📊"), ("Engenharia da Computação", "💻"), 
                ("Matemática", "🧮"), ("Física", "🔭"), 
                ("Engenharia Elétrica", "⚡")  # Substituição aqui
            ]
            st.write("Selecione até **3 cursos** de Exatas que mais te interessam:")
        
        selecoes = []
        cols = st.columns(3)
        for i, (curso, icone) in enumerate(cursos):
            with cols[i % 3]:
                if st.checkbox(f"{icone} {curso}", key=f"curso_{i}"):
                    selecoes.append(curso)
                    if len(selecoes) >= 3:
                        st.warning("Limite de 3 cursos atingido.")
                        break
        
        if selecoes:
            # Dados de cursos para clustering (one-hot encoding)
            if perfil == "Humanas":
                X_cursos = np.array([
                    [1, 0, 0, 0, 0],  # Letras
                    [0, 1, 0, 0, 0],  # História
                    [0, 0, 1, 0, 0],  # Direito
                    [0, 0, 0, 1, 0],  # Psicologia
                    [0, 0, 0, 0, 1]   # Artes
                ])
            else:
                X_cursos = np.array([
                    [1, 0, 0, 0, 0],  # Estatística
                    [0, 1, 0, 0, 0],  # Eng. Computação
                    [0, 0, 1, 0, 0],  # Matemática
                    [0, 0, 0, 1, 0],  # Física
                    [0, 0, 0, 0, 1]   # Eng. Elétrica
                ])
            
            # K-means para cursos (3 clusters)
            kmeans_cursos = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans_cursos.fit(X_cursos)
            
            # Visualização
            pca_cursos = PCA(n_components=2)
            X_2d_cursos = pca_cursos.fit_transform(X_cursos)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            cores_cursos = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # Cores distintas
            formas_cursos = ["o", "s", "D"]  # Círculo, quadrado, losango
            
            for i in range(len(X_cursos)):
                cluster = kmeans_cursos.labels_[i]
                ax2.scatter(
                    X_2d_cursos[i, 0], X_2d_cursos[i, 1],
                    marker=formas_cursos[cluster],
                    color=cores_cursos[cluster],
                    s=150,
                    edgecolor="black",
                    label=f"{cursos[i][1]} {cursos[i][0]}"
                )
            
            ax2.set_title("Agrupamento de Cursos por Similaridade", pad=20)
            ax2.set_xlabel("Componente Principal 1")
            ax2.set_ylabel("Componente Principal 2")
            ax2.grid(True, linestyle="--", alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            st.pyplot(fig2, bbox_inches="tight")
            
            # Feedback personalizado
            st.success(f"**Cursos selecionados:** {', '.join(selecoes)}")
            if "Estatística" in selecoes:
                st.balloons()
                st.markdown("""
                🎉 **Você tem perfil para Estatística!**  
                📈 Um campo que combina matemática, tecnologia e tomada de decisões.  
                🔍 Explore como a Estatística transforma dados em insights poderosos!
                """)
                
                # Adicionando descrição dos cursos (tooltip alternativo)
                with st.expander("ℹ️ Sobre os cursos de Exatas"):
                    st.markdown("""
                    - **📊 Estatística**: Análise de dados e modelagem matemática.  
                    - **💻 Eng. Computação**: Desenvolvimento de software e hardware.  
                    - **🧮 Matemática**: Fundamentos teóricos e abstração.  
                    - **🔭 Física**: Leis fundamentais do universo.  
                    - **⚡ Eng. Elétrica**: Sistemas de energia e eletrônicos.  
                    """)
