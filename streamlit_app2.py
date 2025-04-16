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
            # Classificação usando K-means
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
            st.session_state.segunda_etapa_respostas = [False] * 10  # 10 características
            st.rerun()

# --- SEGUNDA ETAPA (SIMPLIFICADA) ---
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as 5 características que mais combinam com você")
    
    # Características simplificadas para ensino médio
    caracteristicas = {
        "Exatas": [
            "Gosto de trabalhar com números e estatísticas",
            "Tenho facilidade com matemática",
            "Curto tecnologia e computadores",
            "Gosto de resolver problemas lógicos",
            "Me interesso por como as coisas funcionam",
            "Prefiro coisas concretas e objetivas",
            "Gosto de construir e criar coisas",
            "Tenho curiosidade sobre ciências",
            "Gosto de jogos de estratégia",
            "Prefiro exatidão a interpretações"
        ],
        "Humanas": [
            "Gosto de ler e escrever",
            "Tenho facilidade em me expressar",
            "Me interesso por comportamento humano",
            "Gosto de debater ideias",
            "Tenho sensibilidade artística",
            "Me interesso por questões sociais",
            "Gosto de história e cultura",
            "Tenho facilidade com idiomas",
            "Prefiro trabalhar com pessoas",
            "Gosto de interpretar textos"
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
            # Mapeamento curso-características simplificado
            cursos_map = {
                "Exatas": {
                    "Estatística": [0, 1, 5, 9],
                    "Matemática": [1, 3, 6, 9],
                    "Engenharias": [2, 4, 6, 7],
                    "Ciência da Computação": [2, 3, 8, 9],
                    "Física": [1, 4, 7, 9]
                },
                "Humanas": {
                    "Direito": [1, 3, 5, 9],
                    "História": [0, 6, 7, 9],
                    "Letras": [0, 1, 7, 9],
                    "Psicologia": [2, 3, 5, 8],
                    "Artes": [0, 4, 6, 8],
                    "Ciências Sociais": [3, 5, 6, 8]
                }
            }[st.session_state.perfil]
            
            # Calcular similaridade usando clustering
            # Criar vetor de características selecionadas (1=selecionado, 0=não selecionado)
            vetor_usuario = np.array([1 if carac in selecoes else 0 for carac in caracteristicas])
            
            # Criar dados de treino baseados nos cursos
            dados_treino = []
            cursos_lista = []
            for curso, caracs in cursos_map.items():
                # Criar 3 exemplos por curso (com pequenas variações)
                for _ in range(3):
                    vetor = np.zeros(len(caracteristicas))
                    # Ativar características principais do curso
                    for idx in caracs:
                        vetor[idx] = 1
                    # Adicionar pequeno ruído
                    vetor += np.random.normal(0, 0.1, len(vetor))
                    dados_treino.append(vetor)
                    cursos_lista.append(curso)
            
            dados_treino = np.array(dados_treino)
            
            # Aplicar PCA para visualização 2D
            pca = PCA(n_components=2)
            dados_2d = pca.fit_transform(dados_treino)
            usuario_2d = pca.transform(vetor_usuario.reshape(1, -1))
            
            # Clusterizar os cursos
            kmeans = KMeans(n_clusters=len(cursos_map), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(dados_2d)
            
            # Determinar qual cluster o usuário pertence
            cluster_usuario = kmeans.predict(usuario_2d)[0]
            
            # Encontrar o curso mais comum no cluster do usuário
            cursos_no_cluster = [cursos_lista[i] for i, c in enumerate(clusters) if c == cluster_usuario]
            from collections import Counter
            curso_ideal = Counter(cursos_no_cluster).most_common(1)[0][0]
            
            # --- VISUALIZAÇÃO ---
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Cores para os cursos
            cores = plt.cm.get_cmap('tab10', len(cursos_map))
            curso_para_cor = {curso: i for i, curso in enumerate(cursos_map.keys())}
            
            # Plotar pontos dos cursos
            for i, (x, y) in enumerate(dados_2d):
                curso = cursos_lista[i]
                ax.scatter(x, y, color=cores(curso_para_cor[curso]), 
                          label=curso if i < len(cursos_map) else "", s=100, alpha=0.7)
            
            # Plotar usuário
            ax.scatter(usuario_2d[0, 0], usuario_2d[0, 1], color=cores(curso_para_cor[curso_ideal]), 
                      marker="*", s=300, edgecolor="black", label="Você")
            
            # Configurações do gráfico
            ax.set_title("Sua Proximidade com os Cursos", pad=20)
            ax.set_xlabel("Componente Principal 1")
            ax.set_ylabel("Componente Principal 2")
            ax.legend(bbox_to_anchor=(1.05, 1))
            ax.grid(True, linestyle="--", alpha=0.3)
            
            st.pyplot(fig)
            
            # --- RESULTADO FINAL ---
            st.balloons()
            
            emoji_curso = {
                "Estatística": "📊", "Matemática": "🧮", 
                "Engenharias": "⚙️", "Ciência da Computação": "💻",
                "Física": "🔭", "Direito": "⚖️", "História": "🏛️", 
                "Letras": "📖", "Psicologia": "🧠", "Artes": "🎨",
                "Ciências Sociais": "👥"
            }.get(curso_ideal, "🎓")
            
            st.success(f"""
            **Resultado Final:**
            
            🎯 **Você tem perfil de {st.session_state.perfil}** e se encaixa melhor em:
            {emoji_curso} **{curso_ideal}**
            
            **Características que mais combinam:**
            """)
            
            # Listar características selecionadas que são relevantes para o curso
            caracs_relevantes = [
                carac for i, carac in enumerate(caracteristicas) 
                if (i in cursos_map[curso_ideal] and st.session_state.segunda_etapa_respostas[i])
            ]
            
            for carac in caracs_relevantes[:5]:  # Mostrar no máximo 5
                st.write(f"- {carac}")

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()    
