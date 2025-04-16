import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuração da página
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico")

# Variáveis de sessão
if 'etapa' not in st.session_state:
    st.session_state.etapa = 1
    st.session_state.perfil = None
    st.session_state.respostas = None
    st.session_state.segunda_etapa_respostas = None

# --- PRIMEIRA ETAPA (ORIGINAL FUNCIONANDO) ---
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
            # Classificação original usando K-means
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
            st.session_state.respostas = respostas
            st.session_state.etapa = 2
            st.session_state.segunda_etapa_respostas = [False] * 12  # 12 características
            st.rerun()

# --- SEGUNDA ETAPA (CORRIGIDA) ---
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as características que mais combinam com você")
    
    # Características por área - Definições mais específicas e exclusivas
    caracteristicas = {
        "Exatas": [
            "📊 Modelagem estatística avançada",  # Estatística
            "🧮 Teoria matemática abstrata",      # Matemática
            "⚡ Projeto de circuitos integrados",  # Eng. Elétrica
            "🏗️ Cálculo estrutural de pontes",    # Eng. Civil
            "💻 Desenvolvimento de algoritmos",   # Ciência da Computação
            "📈 Análise preditiva de dados",      # Estatística
            "🔢 Equações diferenciais parciais",  # Matemática
            "📐 Dinâmica de fluidos computacional", # Eng. Mecânica
            "🌐 Arquitetura de redes complexas",   # Ciência da Computação
            "🧪 Física quântica aplicada",        # Engenharias/Física
            "🤖 Controle de sistemas autônomos",  # Eng. Controle
            "📉 Visualização científica de dados"  # Estatística
        ],
        "Humanas": [
            "⚖️ Direito constitucional",         # Direito
            "📜 Paleografia e documentos antigos", # História
            "📖 Teoria literária crítica",        # Letras
            "🧠 Neuropsicologia cognitiva",       # Psicologia
            "🗣️ Mediação de conflitos",          # Psicologia/Direito
            "🎨 Curadoria de exposições",         # Artes
            "🌍 Etnografia cultural",             # Antropologia
            "✍️ Redação acadêmica",               # Vários
            "🏛️ Arqueologia clássica",           # História
            "👥 Psicologia social",               # Psicologia
            "💬 Oratória persuasiva",             # Comunicação
            "📝 Elaboração de contratos"          # Direito
        ]
    }[st.session_state.perfil]

    # Atualizar seleções mantendo estado
    cols = st.columns(2)
    selecoes = []
    for i, carac in enumerate(caracteristicas):
        with cols[i % 2]:
            checked = st.checkbox(carac, key=f"carac_{i}", 
                                value=st.session_state.segunda_etapa_respostas[i])
            st.session_state.segunda_etapa_respostas[i] = checked
            if checked:
                selecoes.append(carac)

    if st.button("🎯 Descobrir meu curso ideal"):
        if len(selecoes) != 5:
            st.warning("Selecione exatamente 5 características!")
        else:
            # Mapeamento curso-características (mais específico)
            cursos_map = {
                "Exatas": {
                    "Estatística": [0, 5, 11],  # Características exclusivas
                    "Matemática": [1, 6],
                    "Engenharia Elétrica": [2],
                    "Engenharia Civil": [3],
                    "Ciência da Computação": [4, 8],
                    "Engenharia Mecânica": [7],
                    "Física": [9],
                    "Engenharia de Controle": [10]
                },
                "Humanas": {
                    "Direito": [0, 11],
                    "História": [1, 8],
                    "Letras": [2],
                    "Psicologia": [3, 4, 9],
                    "Artes": [5],
                    "Antropologia": [6],
                    "Comunicação": [10],
                    "Ciências Sociais": [7]
                }
            }[st.session_state.perfil]
            
            # Calcular pontuação para cada curso
            scores = {}
            for curso, idx_caracs in cursos_map.items():
                score = sum(1 for idx in idx_caracs 
                           if caracteristicas[idx] in selecoes)
                scores[curso] = score
            
            # Curso ideal é o com maior pontuação (desempate por ordem de preferência)
            curso_ideal = max(scores.items(), key=lambda x: (x[1], -list(scores.keys()).index(x[0])))[0]
            
            # --- VISUALIZAÇÃO MELHORADA ---
            # Gerar pontos para os cursos (3 por curso)
            num_cursos = len(cursos_map)
            pontos_curso = {}
            for i, curso in enumerate(cursos_map.keys()):
                # Posição base no eixo X + pequena variação
                x_base = i
                pontos_curso[curso] = np.column_stack([
                    np.random.normal(x_base, 0.1, size=3),  # Posição X
                    np.random.normal(0, 0.1, size=3)        # Posição Y
                ])
            
            # Posição do usuário (próxima ao centroide do curso ideal)
            centroide = np.mean(pontos_curso[curso_ideal], axis=0)
            user_pos = centroide + np.array([0, 0.2])  # Posiciona acima do cluster
            
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            cores = plt.cm.get_cmap('tab10', num_cursos)
            
            for i, (curso, pontos) in enumerate(pontos_curso.items()):
                ax.scatter(
                    pontos[:, 0], pontos[:, 1],
                    color=cores(i),
                    s=100,
                    label=f"{curso} ({scores[curso]})",
                    alpha=0.8,
                    edgecolor='black'
                )
            
            # Plotar usuário
            ax.scatter(
                user_pos[0], user_pos[1],
                color=cores(list(cursos_map.keys()).index(curso_ideal)),
                marker="*",
                s=300,
                edgecolor="black",
                label=f"Você → {curso_ideal}"
            )
            
            # Configurações do gráfico
            ax.set_title("Sua Proximidade com os Cursos", pad=20)
            ax.set_xticks(range(num_cursos))
            ax.set_xticklabels(cursos_map.keys(), rotation=45, ha='right')
            ax.set_yticks([])
            ax.legend(bbox_to_anchor=(1.05, 1))
            ax.grid(True, linestyle="--", alpha=0.3)
            
            st.pyplot(fig)
            
            # --- RESULTADO FINAL ---
            st.balloons()
            
            emoji_curso = {
                "Estatística": "📊", "Matemática": "🧮", 
                "Engenharia Elétrica": "⚡", "Engenharia Civil": "🏗️",
                "Ciência da Computação": "💻", "Engenharia Mecânica": "⚙️",
                "Física": "🔭", "Engenharia de Controle": "🤖",
                "Direito": "⚖️", "História": "🏛️", "Letras": "📖",
                "Psicologia": "🧠", "Artes": "🎨", "Antropologia": "🌍",
                "Comunicação": "💬", "Ciências Sociais": "👥"
            }.get(curso_ideal, "🎓")
            
            st.success(f"""
            **Resultado Final:**
            
            🎯 **Você tem perfil de {st.session_state.perfil}** e se encaixa melhor em:
            {emoji_curso} **{curso_ideal}**
            
            **Características que mais combinam:**
            """)
            
            # Listar apenas características que contribuíram para o curso
            caracs_correspondentes = [
                carac for carac in selecoes 
                if any(carac == caracteristicas[idx] for idx in cursos_map[curso_ideal])
            ]
            
            for carac in caracs_correspondentes:
                st.write(f"- {carac}")

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()    
