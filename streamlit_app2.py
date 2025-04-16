import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

# Configuração da página
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico")

# Variáveis de sessão
if 'etapa' not in st.session_state:
    st.session_state.etapa = 1
    st.session_state.perfil = None
    st.session_state.respostas = None

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
            st.session_state.respostas = respostas
            st.session_state.etapa = 2
            st.rerun()

# --- SEGUNDA ETAPA ---
elif st.session_state.etapa == 2:
    # Classificação inicial (Humanas/Exatas)
    X_novo = np.array(st.session_state.respostas).reshape(1, -1)
    grupo_humanas = np.array([
        [1,0,1,0,1,0,1,0,1,0], [0,0,1,0,1,0,0,0,1,0],
        [1,0,0,0,1,0,1,0,0,0], [0,0,1,0,0,0,1,0,1,0],
        [1,0,1,0,0,0,0,0,1,0]
    ])
    grupo_exatas = np.array([
        [0,1,0,1,0,1,0,1,0,1], [1,1,0,1,0,1,0,0,0,1],
        [0,1,0,0,0,1,0,1,0,0], [1,0,0,1,0,1,0,1,0,0],
        [0,1,0,1,0,0,0,1,0,1]
    ])
    X_treino = np.vstack((grupo_humanas, grupo_exatas))
    
    kmeans_geral = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_treino)
    perfil = "Humanas" if kmeans_geral.predict(X_novo)[0] == 0 else "Exatas"
    st.session_state.perfil = perfil

    st.success(f"Perfil principal: **{perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as características que mais combinam com você")
    
    # Características por área (sem revelar os cursos)
    caracteristicas = {
        "Exatas": [
            "📊 Analisar dados e estatísticas",
            "🧮 Resolver problemas matemáticos complexos",
            "📈 Trabalhar com probabilidades e previsões",
            "⚡ Projetar sistemas elétricos e circuitos",
            "🏗️ Projetar estruturas e construções",
            "💻 Desenvolver algoritmos e programas",
            "🔢 Criar modelos matemáticos avançados",
            "🌉 Calcular cargas e resistências de materiais",
            "🔧 Projetar máquinas e sistemas mecânicos",
            "📐 Fazer cálculos estruturais precisos"
        ],
        "Humanas": [
            "⚖️ Argumentar e interpretar leis",
            "📜 Analisar documentos históricos",
            "📖 Escrever textos criativos",
            "🧠 Entender comportamentos humanos",
            "🗣️ Mediar conflitos e debates",
            "🎨 Analisar expressões artísticas",
            "🌍 Estudar culturas e sociedades",
            "✍️ Produzir conteúdo literário",
            "🏛️ Interpretar contextos históricos",
            "👥 Trabalhar com dinâmicas de grupo"
        ]
    }[perfil]

    # Seleção por características
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
            # Mapeamento curso-características (oculto)
            cursos_map = {
                "Exatas": {
                    "Estatística": [0, 1, 2, 6],
                    "Engenharia Elétrica": [3],
                    "Engenharia Civil": [4, 7, 9],
                    "Ciência da Computação": [5],
                    "Engenharia Mecânica": [8]
                },
                "Humanas": {
                    "Direito": [0],
                    "História": [1, 8],
                    "Letras": [2, 7],
                    "Psicologia": [3, 9],
                    "Artes": [5]
                }
            }[perfil]
            
            # Pré-processamento
            scaler = StandardScaler()
            
            # Vetor do usuário
            X_usuario = np.array([1 if carac in selecoes else 0 for carac in caracteristicas])
            
            # Gerar dados sintéticos para cada curso (simulando "pessoas" em cada curso)
            X_cursos = []
            labels = []
            for curso, idx_caracs in cursos_map.items():
                # Gerar 5-10 perfis sintéticos para cada curso
                for _ in range(random.randint(5, 10)):
                    vec = np.zeros(len(caracteristicas))
                    # Garantir que as características principais estejam sempre presentes
                    for idx in idx_caracs:
                        vec[idx] = 1
                    # Adicionar algumas variações aleatórias
                    for _ in range(random.randint(1, 3)):
                        vec[random.choice(idx_caracs)] = 0  # Remover alguma característica
                        vec[random.randint(0, len(caracteristicas)-1)] = 1  # Adicionar aleatório
                    X_cursos.append(vec)
                    labels.append(curso)
            
            X_cursos = np.array(X_cursos)
            
            # Adicionar o usuário
            X_combined = np.vstack((X_cursos, X_usuario))
            X_scaled = scaler.fit_transform(X_combined)
            
            # K-means ajustado
            kmeans = KMeans(
                n_clusters=len(cursos_map),
                init='k-means++',
                random_state=42,
                n_init=20
            ).fit(X_scaled[:-1])  # Treina apenas nos dados sintéticos
            
            # Predição
            cluster_usuario = kmeans.predict(X_scaled[-1].reshape(1, -1))[0]
            curso_ideal = list(cursos_map.keys())[cluster_usuario]
            
            # PCA para visualização
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            
            # --- Gráfico 1: Perfil Geral (Humanas/Exatas) ---
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            
            # Transformar os dados da primeira etapa
            X_vis_geral = np.vstack((X_treino, X_novo))
            pca_geral = PCA(n_components=2)
            X_2d_geral = pca_geral.fit_transform(X_vis_geral)
            
            # Plot dados de treino
            for i in range(len(X_treino)):
                cluster = kmeans_geral.labels_[i] if i < len(X_treino) else -1
                ax1.scatter(
                    X_2d_geral[i, 0], X_2d_geral[i, 1],
                    marker="o" if i < len(grupo_humanas) else "s",
                    color="blue" if i < len(grupo_humanas) else "red",
                    alpha=0.6
                )
            
            # Plot usuário
            ax1.scatter(
                X_2d_geral[-1, 0], X_2d_geral[-1, 1],
                marker="*",
                color="green",
                s=200,
                edgecolor="black",
                label="Você"
            )
            
            ax1.set_title("1. Seu Perfil Geral (Humanas vs Exatas)", fontsize=12)
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.3)
            
            # --- Gráfico 2: Cursos Específicos ---
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            cores = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#9B59B6", "#2ECC71"]
            
            # Plot cada curso com múltiplos pontos
            for i, curso in enumerate(cursos_map.keys()):
                indices = [j for j, label in enumerate(labels) if label == curso]
                ax2.scatter(
                    X_2d[indices, 0], X_2d[indices, 1],
                    color=cores[i],
                    marker=["o", "s", "D", "^", "p"][i],
                    alpha=0.7,
                    s=100,
                    label=curso,
                    edgecolor='black',
                    linewidth=0.5
                )
            
            # Plot usuário
            ax2.scatter(
                X_2d[-1, 0], X_2d[-1, 1],
                marker="*",
                color="gold",
                s=300,
                edgecolor="black",
                label="Você"
            )
            
            ax2.set_title("2. Curso Ideal Baseado nas Características", fontsize=12)
            ax2.legend(bbox_to_anchor=(1.05, 1))
            ax2.grid(True, linestyle="--", alpha=0.3)
            
            # Mostrar ambos os gráficos
            st.pyplot(fig1)
            st.pyplot(fig2)
            
            st.balloons()
            
            # Resultado com descrição
            emoji_curso = {
                "Estatística": "📊",
                "Engenharia Elétrica": "⚡",
                "Engenharia Civil": "🏗️",
                "Ciência da Computação": "💻",
                "Engenharia Mecânica": "⚙️",
                "Direito": "⚖️",
                "História": "🏛️",
                "Letras": "📖",
                "Psicologia": "🧠",
                "Artes": "🎨"
            }.get(curso_ideal, "🎓")
            
            st.success(f"{emoji_curso} **Curso ideal:** {curso_ideal}")
            
            # Descrição detalhada
            descricoes = {
                "Estatística": "Análise de dados, probabilidade e modelagem matemática para tomada de decisões.",
                "Engenharia Civil": "Projeto, construção e manutenção de infraestruturas e edificações.",
                "Ciência da Computação": "Desenvolvimento de algoritmos, sistemas computacionais e inteligência artificial."
            }
            
            if curso_ideal in descricoes:
                st.info(f"**Sobre {curso_ideal}:** {descricoes[curso_ideal]}")

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()   
