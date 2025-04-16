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
        [1,0,0,0,1,0,1,0,0,0]
    ])
    grupo_exatas = np.array([
        [0,1,0,1,0,1,0,1,0,1], [1,1,0,1,0,1,0,0,0,1],
        [0,1,0,0,0,1,0,1,0,0]
    ])
    X_treino = np.vstack((grupo_humanas, grupo_exatas))
    
    kmeans_geral = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_treino)
    perfil = "Humanas" if kmeans_geral.predict(X_novo)[0] == 0 else "Exatas"
    st.session_state.perfil = perfil

    st.success(f"Perfil principal: **{perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as características que mais combinam com você")
    
    # Características por área (sem revelar cursos)
    caracteristicas = {
        "Exatas": [
            "📊 Analisar dados e estatísticas",
            "🧮 Resolver problemas matemáticos",
            "📈 Trabalhar com probabilidades",
            "⚡ Projetar sistemas elétricos",
            "🏗️ Projetar estruturas e construções",
            "💻 Desenvolver algoritmos"
        ],
        "Humanas": [
            "⚖️ Argumentar e interpretar leis",
            "📜 Analisar documentos históricos",
            "📖 Escrever textos criativos",
            "🧠 Entender comportamentos humanos",
            "🗣️ Mediar conflitos e debates",
            "🎨 Analisar expressões artísticas"
        ]
    }[perfil]

    # Seleção por características
    st.write("**Selecione 3 características:**")
    cols = st.columns(2)
    selecoes = []
    
    for i, carac in enumerate(caracteristicas):
        with cols[i % 2]:
            if st.checkbox(carac, key=f"carac_{i}"):
                selecoes.append(carac)
                if len(selecoes) >= 3:
                    break  # Limita a 3 seleções

    if st.button("🎯 Descobrir meu curso ideal"):
        if len(selecoes) < 3:
            st.warning("Selecione exatamente 3 características!")
        else:
            # Mapeamento curso-características (oculto)
            cursos_map = {
                "Exatas": {
                    "Estatística": [0, 1, 2],  # Índices das características
                    "Engenharia Elétrica": [3],
                    "Engenharia Civil": [4],
                    "Ciência da Computação": [5]
                },
                "Humanas": {
                    "Direito": [0],
                    "História": [1],
                    "Letras": [2],
                    "Psicologia": [3]
                }
            }[perfil]
            
            # Pré-processamento
            scaler = StandardScaler()
            
            # Vetor do usuário (one-hot)
            X_usuario = np.array([1 if carac in selecoes else 0 for carac in caracteristicas])
            
            # Gerar 3 pontos sintéticos por curso (bem definidos)
            X_cursos = []
            labels = []
            for curso, idx_caracs in cursos_map.items():
                for _ in range(3):  # 3 pontos por curso
                    vec = np.zeros(len(caracteristicas))
                    # Características principais sempre presentes
                    for idx in idx_caracs:
                        vec[idx] = 1
                    # Adicionar pequena variação
                    if len(idx_caracs) < len(caracteristicas):
                        vec[np.random.choice([i for i in range(len(caracteristicas)) if i not in idx_caracs])] = 1
                    X_cursos.append(vec)
                    labels.append(curso)
            
            X_cursos = np.array(X_cursos)
            
            # Combinar dados
            X_combined = np.vstack((X_cursos, X_usuario))
            X_scaled = scaler.fit_transform(X_combined)
            
            # K-means ajustado (agora com inicialização personalizada)
            kmeans = KMeans(
                n_clusters=len(cursos_map),
                init=X_scaled[:len(cursos_map)],  # Inicializa nos centros dos cursos
                random_state=42,
                n_init=1
            ).fit(X_scaled[:-1])  # Treina apenas nos dados de referência
            
            # Determinar curso por proximidade ao centróide
            distancias = kmeans.transform(X_scaled[-1].reshape(1, -1))
            curso_ideal = list(cursos_map.keys())[np.argmin(distancias)]
            
            # PCA para visualização
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            
            # --- Gráfico 1: Perfil Geral ---
            pca_geral = PCA(n_components=2)
            X_2d_geral = pca_geral.fit_transform(np.vstack((X_treino, X_novo)))
            
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            for i in range(len(X_treino)):
                ax1.scatter(
                    X_2d_geral[i, 0], X_2d_geral[i, 1],
                    color="blue" if i < len(grupo_humanas) else "red",
                    marker="o" if i < len(grupo_humanas) else "s",
                    alpha=0.6
                )
            ax1.scatter(X_2d_geral[-1, 0], X_2d_geral[-1, 1], color="gold", marker="*", s=200, label="Você")
            ax1.set_title("1. Seu Perfil Geral (Humanas vs Exatas)")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.3)
            
            # --- Gráfico 2: Cursos Específicos ---
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            cores = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#9B59B6"]
            marcadores = ["o", "s", "D", "^"]
            
            for i, curso in enumerate(cursos_map.keys()):
                indices = [j for j, label in enumerate(labels) if label == curso]
                ax2.scatter(
                    X_2d[indices, 0], X_2d[indices, 1],
                    color=cores[i],
                    marker=marcadores[i],
                    s=100,
                    label=curso,
                    alpha=0.8,
                    edgecolor='black'
                )
            
            ax2.scatter(X_2d[-1, 0], X_2d[-1, 1], color="gold", marker="*", s=300, label="Você")
            ax2.set_title("2. Proximidade aos Cursos (3 pontos por curso)")
            ax2.legend(bbox_to_anchor=(1.05, 1))
            ax2.grid(True, linestyle="--", alpha=0.3)
            
            # Exibir gráficos
            st.pyplot(fig1)
            st.pyplot(fig2)
            
            st.balloons()
            
            # Resultado com descrição
            emoji_curso = {
                "Estatística": "📊",
                "Engenharia Elétrica": "⚡",
                "Engenharia Civil": "🏗️",
                "Ciência da Computação": "💻",
                "Direito": "⚖️",
                "História": "🏛️",
                "Letras": "📖",
                "Psicologia": "🧠"
            }.get(curso_ideal, "🎓")
            
            st.success(f"{emoji_curso} **Curso ideal:** {curso_ideal}")
            
            # Explicação da decisão
            st.info(f"**Por que {curso_ideal}?** Você selecionou características que são essenciais para este curso:" + 
                   "".join([f"\n- {carac}" for carac in selecoes if carac in [caracteristicas[i] for i in cursos_map[curso_ideal]]]))

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()   
