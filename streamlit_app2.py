import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico")

# Variáveis de sessão
if 'etapa' not in st.session_state:
    st.session_state.etapa = 1
    st.session_state.perfil = None

# --- PRIMEIRA ETAPA ---
if st.session_state.etapa == 1:
    st.write("**Parte 1/2:** Marque os conteúdos com que você mais se identifica:")
    
    itens = [
        {"texto": "Escrever poemas ou crônicas", "valor": 0},
        {"texto": "Resolver desafios de programação", "valor": 1},
        {"texto": "Debater sobre filosofia/sociologia", "valor": 0},
        {"texto": "Projetar experimentos científicos", "valor": 1},
        {"texto": "Analisar obras de arte", "valor": 0},
        {"texto": "Desenvolver fórmulas matemáticas", "valor": 1},
        {"texto": "Ler sobre política internacional", "valor": 0},
        {"texto": "Estudar novas tecnologias", "valor": 1},
        {"texto": "Interpretar textos literários", "valor": 0},
        {"texto": "Trabalhar com cálculos complexos", "valor": 1}
    ]
    
    respostas = [0] * len(itens)
    for i, item in enumerate(itens):
        if st.checkbox(item["texto"], key=f"item_{i}"):
            respostas[i] = 1

    if st.button("🔎 Avançar para a Parte 2"):
        if sum(respostas) < 3:
            st.warning("Selecione pelo menos 3 conteúdos!")
        else:
            X_novo = np.array(respostas).reshape(1, -1)
            grupo_humanas = np.array([
                [1,0,1,0,1,0,1,0,1,0], [0,0,1,0,1,0,0,0,1,0],
                [1,0,0,0,1,0,1,0,0,0]
            ])
            grupo_exatas = np.array([
                [0,1,0,1,0,1,0,1,0,1], [1,1,0,1,0,1,0,0,0,1],
                [0,1,0,0,0,1,0,1,0,0]
            ])
            X_treino = np.vstack((grupo_humanas, grupo_exatas))
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_treino)
            st.session_state.perfil = "Humanas" if kmeans.predict(X_novo)[0] == 0 else "Exatas"
            st.session_state.etapa = 2
            st.rerun()

# --- SEGUNDA ETAPA ---
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as características que mais combinam com você")
    
    # Dicionário de características por curso (sem mostrar os nomes dos cursos)
    caracteristicas = {
        "Exatas": [
            "📊 Analisar dados e estatísticas",
            "🧮 Resolver problemas matemáticos complexos",
            "📈 Trabalhar com probabilidades e previsões",
            "⚡ Projetar sistemas elétricos e circuitos",
            "🏗️ Projetar estruturas e construções",  # Engenharia Civil
            "💻 Desenvolver algoritmos e programas",
            "🔢 Criar modelos matemáticos avançados",
            "🌉 Calcular cargas e resistências de materiais"  # Engenharia Civil
        ],
        "Humanas": [
            "⚖️ Argumentar e interpretar leis",
            "📜 Analisar documentos históricos",
            "📖 Escrever textos criativos",
            "🧠 Entender comportamentos humanos",
            "🗣️ Mediar conflitos e debates",
            "🎨 Analisar expressões artísticas",
            "🌍 Estudar culturas e sociedades",
            "✍️ Produzir conteúdo literário"
        ]
    }[st.session_state.perfil]

    # Seleção por características (agrupadas em colunas)
    st.write("**Selecione 3 a 5 características:**")
    cols = st.columns(2)
    selecoes = []
    
    for i, carac in enumerate(caracteristicas):
        with cols[i % 2]:
            if st.checkbox(carac, key=f"carac_{i}"):
                selecoes.append(carac)

    if st.button("🎯 Descobrir meu curso ideal"):
        if len(selecoes) < 3:
            st.warning("Selecione pelo menos 3 características!")
        else:
            # Mapeamento curso-características (agora oculto ao usuário)
            cursos_map = {
                "Exatas": {
                    "Estatística": [0, 1, 2],  # Índices das características
                    "Engenharia Elétrica": [3],
                    "Engenharia Civil": [4, 7],  # Adicionado
                    "Ciência da Computação": [5],
                    "Matemática": [6]
                },
                "Humanas": {
                    "Direito": [0],
                    "História": [1],
                    "Letras": [2, 7],
                    "Psicologia": [3],
                    "Artes": [5]
                }
            }[st.session_state.perfil]
            
            # Pré-processamento
            scaler = StandardScaler()
            
            # Vetor do usuário (one-hot)
            X_usuario = np.array([1 if carac in selecoes else 0 for carac in caracteristicas])
            
            # Matriz de referência (cursos)
            X_cursos = []
            for curso, idx_caracs in cursos_map.items():
                vec = np.zeros(len(caracteristicas))
                for idx in idx_caracs:
                    vec[idx] = 1
                X_cursos.append(vec)
            X_cursos = np.array(X_cursos)
            
            # Normalização
            X_combined = np.vstack((X_cursos, X_usuario))
            X_scaled = scaler.fit_transform(X_combined)
            
            # K-means com inicialização personalizada
            kmeans = KMeans(
                n_clusters=len(cursos_map),
                init=X_scaled[:len(cursos_map)],  # Inicializa com os centros dos cursos
                random_state=42,
                n_init=1
            ).fit(X_scaled[:-1])  # Ajusta apenas nos dados de referência
            
            # Predição
            cluster_usuario = kmeans.predict(X_scaled[-1].reshape(1, -1))[0]
            curso_ideal = list(cursos_map.keys())[cluster_usuario]
            
            # PCA para visualização (2D)
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            cores = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#9B59B6", "#2ECC71"]
            formas = ["o", "s", "D", "^", "p"]
            
            # Plot cursos de referência
            for i in range(len(X_cursos)):
                ax.scatter(
                    X_2d[i, 0], X_2d[i, 1],
                    marker=formas[i],
                    color=cores[i],
                    s=200,
                    edgecolor="black",
                    label=list(cursos_map.keys())[i]
                )
            
            # Plot usuário
            ax.scatter(
                X_2d[-1, 0], X_2d[-1, 1],
                marker="*",
                color="gold",
                s=400,
                edgecolor="black",
                label="Você"
            )
            
            ax.set_title("Seu perfil comparado aos cursos", fontsize=14)
            ax.set_xlabel("Componente Principal 1", fontsize=10)
            ax.set_ylabel("Componente Principal 2", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(bbox_to_anchor=(1.3, 1))
            
            st.pyplot(fig)
            st.balloons()
            
            # Resultado com emoji do curso
            emoji_curso = {
                "Estatística": "📊",
                "Engenharia Elétrica": "⚡",
                "Engenharia Civil": "🏗️",
                "Ciência da Computação": "💻",
                "Matemática": "🧮",
                "Direito": "⚖️",
                "História": "🏛️",
                "Letras": "📖",
                "Psicologia": "🧠",
                "Artes": "🎨"
            }.get(curso_ideal, "🎓")
            
            st.success(f"{emoji_curso} **Curso ideal:** {curso_ideal}")
            
            # Descrição especial para Engenharia Civil
            if curso_ideal == "Engenharia Civil":
                st.markdown("""
                **🏗️ Características do Engenheiro Civil:**  
                - Projetar e supervisionar construções  
                - Calcular estruturas e materiais  
                - Resolver problemas urbanos e ambientais  
                - Trabalhar com infraestrutura e transporte  
                """)

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()   
