import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    
    # Características por área (12 opções)
    caracteristicas = {
        "Exatas": [
            "📊 Analisar dados estatísticos",
            "🧮 Resolver problemas matemáticos complexos",
            "📈 Trabalhar com probabilidades e previsões",
            "⚡ Projetar sistemas elétricos/circuitos",
            "🏗️ Calcular estruturas e resistência de materiais",
            "💻 Programar algoritmos complexos",
            "🔢 Desenvolver modelos matemáticos",
            "📐 Realizar cálculos estruturais precisos",
            "📉 Interpretar gráficos e tendências",
            "🧪 Realizar experimentos quantitativos",
            "🤖 Trabalhar com inteligência artificial",
            "🌐 Desenvolver sistemas computacionais"
        ],
        "Humanas": [
            "⚖️ Argumentar e interpretar leis",
            "📜 Analisar documentos históricos",
            "📖 Escrever textos criativos/literários",
            "🧠 Estudar comportamentos humanos",
            "🗣️ Mediar conflitos e debates",
            "🎨 Analisar expressões artísticas",
            "🌍 Estudar culturas e sociedades",
            "✍️ Produzir conteúdo escrito",
            "🏛️ Interpretar contextos históricos",
            "👥 Trabalhar com dinâmicas de grupo",
            "💬 Desenvolver comunicação interpessoal",
            "📝 Redigir documentos formais"
        ]
    }[perfil]

    # Seleção por características (agora 5 seleções)
    st.write("**Selecione 5 características que mais combinam com você:**")
    cols = st.columns(2)
    selecoes = []
    
    for i, carac in enumerate(caracteristicas):
        with cols[i % 2]:
            if st.checkbox(carac, key=f"carac_{i}"):
                selecoes.append(carac)
                if len(selecoes) == 5:  # Limite de 5 seleções
                    break

    if st.button("🎯 Descobrir meu curso ideal"):
        if len(selecoes) < 5:
            st.warning("Selecione exatamente 5 características!")
        else:
            # Mapeamento curso-características (priorizando características essenciais)
            cursos_map = {
                "Exatas": {
                    "Estatística": [0, 1, 2, 6, 8, 9],  # + características exclusivas
                    "Engenharia Elétrica": [3, 5, 10],
                    "Engenharia Civil": [4, 7],
                    "Ciência da Computação": [5, 10, 11],
                    "Matemática Aplicada": [1, 2, 6]
                },
                "Humanas": {
                    "Direito": [0, 7, 11],
                    "História": [1, 8],
                    "Letras": [2, 3, 7],
                    "Psicologia": [3, 4, 9],
                    "Artes": [5, 6]
                }
            }[perfil]
            
            # Pré-processamento
            scaler = StandardScaler()
            
            # Vetor do usuário (one-hot)
            X_usuario = np.array([1 if carac in selecoes else 0 for carac in caracteristicas])
            
            # Gerar dados de referência (3 pontos por curso)
            X_cursos = []
            labels = []
            for curso, idx_caracs in cursos_map.items():
                for _ in range(3):  # 3 pontos por curso
                    vec = np.zeros(len(caracteristicas))
                    # Características principais (garantidas)
                    for idx in idx_caracs[:3]:  # Pelo menos 3 principais
                        vec[idx] = 1
                    # Características secundárias (opcionais)
                    for idx in idx_caracs[3:]:
                        vec[idx] = random.choice([0, 1])  # Alguma variação
                    X_cursos.append(vec)
                    labels.append(curso)
            
            X_cursos = np.array(X_cursos)
            
            # Combinar dados
            X_combined = np.vstack((X_cursos, X_usuario))
            X_scaled = scaler.fit_transform(X_combined)
            
            # --- NOVA LÓGICA: Classificação por Similaridade Direta ---
            # Calcula a sobreposição com cada curso
            scores = {}
            for curso, idx_caracs in cursos_map.items():
                score = sum(X_usuario[idx] for idx in idx_caracs)
                scores[curso] = score
            
            curso_ideal = max(scores.items(), key=lambda x: x[1])[0]
            
            # --- Visualização com PCA ---
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            
            # Gráfico 1: Perfil Geral
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
            
            # Gráfico 2: Cursos Específicos
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            cores = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#9B59B6", "#2ECC71"]
            marcadores = ["o", "s", "D", "^", "p"]
            
            for i, curso in enumerate(cursos_map.keys()):
                indices = [j for j, label in enumerate(labels) if label == curso]
                ax2.scatter(
                    X_2d[indices, 0], X_2d[indices, 1],
                    color=cores[i],
                    marker=marcadores[i],
                    s=100,
                    label=f"{curso} (Score: {scores[curso]})",
                    alpha=0.8,
                    edgecolor='black'
                )
            
            # Destacar curso ideal
            idx_ideal = list(cursos_map.keys()).index(curso_ideal)
            ax2.scatter(
                X_2d[-1, 0], X_2d[-1, 1],
                color=cores[idx_ideal],
                marker="*",
                s=300,
                edgecolor="black",
                label=f"Você → {curso_ideal}"
            )
            
            ax2.set_title("2. Similaridade com Cursos (3 pontos por curso)")
            ax2.legend(bbox_to_anchor=(1.35, 1))
            ax2.grid(True, linestyle="--", alpha=0.3)
            
            # Exibir gráficos
            st.pyplot(fig1)
            st.pyplot(fig2)
            
            st.balloons()
            
            # Resultado com explicação
            emoji_curso = {
                "Estatística": "📊",
                "Engenharia Elétrica": "⚡",
                "Engenharia Civil": "🏗️",
                "Ciência da Computação": "💻",
                "Matemática Aplicada": "🧮",
                "Direito": "⚖️",
                "História": "🏛️",
                "Letras": "📖",
                "Psicologia": "🧠",
                "Artes": "🎨"
            }.get(curso_ideal, "🎓")
            
            st.success(f"{emoji_curso} **Curso ideal:** {curso_ideal}")
            
            # Mostrar características correspondentes
            caracs_correspondentes = [carac for i, carac in enumerate(caracteristicas) 
                                     if i in cursos_map[curso_ideal] and X_usuario[i] == 1]
            
            st.info("**Características que mais combinam:**\n" + 
                   "\n".join([f"- {carac}" for carac in caracs_correspondentes]))

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()      
