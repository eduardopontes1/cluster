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
                [1,0,0,0,1,0,1,0,0,0], [1,0,1,0,0,0,1,0,1,0],
                [0,0,1,0,1,0,1,0,0,0], [1,0,0,0,1,0,0,0,1,0]
            ])
            grupo_exatas = np.array([
                [0,1,0,1,0,1,0,1,0,1], [1,1,0,1,0,1,0,0,0,1],
                [0,1,0,0,0,1,0,1,0,0], [0,1,0,1,0,0,0,1,0,1],
                [0,1,1,1,0,1,0,0,0,1], [0,1,0,1,0,1,0,1,0,0]
            ])
            X_treino = np.vstack((grupo_humanas, grupo_exatas))
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_treino)
            st.session_state.perfil = "Humanas" if kmeans.predict(X_novo)[0] == 0 else "Exatas"
            st.session_state.respostas = respostas
            st.session_state.etapa = 2
            st.session_state.segunda_etapa_respostas = [False] * 12  # 12 características
            st.rerun()

# --- SEGUNDA ETAPA (ATUALIZADA) ---
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as 5 características que mais combinam com você")
    
    # Características atualizadas para cursos mais populares
    caracteristicas = {
        "Exatas": [
            "Gosto de analisar dados e estatísticas",
            "Tenho facilidade com cálculos matemáticos",
            "Me interesso por programação e tecnologia",
            "Gosto de resolver problemas práticos",
            "Tenho curiosidade sobre como as coisas funcionam",
            "Prefiro lógica e objetividade",
            "Gosto de construir e projetar coisas",
            "Me interesso por ciências e experimentos",
            "Tenho habilidade com números",
            "Gosto de jogos de raciocínio",
            "Me interesso por inteligência artificial",
            "Tenho facilidade com gráficos e visualizações"
        ],
        "Humanas": [
            "Gosto de ler e escrever textos",
            "Tenho facilidade em me expressar",
            "Me interesso por entender as pessoas",
            "Gosto de debater ideias e opiniões",
            "Tenho sensibilidade artística",
            "Me preocupo com o bem-estar dos outros",
            "Gosto de estudar história e cultura",
            "Tenho facilidade com idiomas",
            "Prefiro trabalhar em grupo",
            "Gosto de interpretar textos e obras",
            "Me interesso por política e sociedade",
            "Tenho habilidade para ajudar os outros"
        ]
    }[st.session_state.perfil]

    # Mapeamento curso-características (5 cursos mais populares por área)
    cursos_map = {
        "Exatas": {
            "Estatística": [0, 1, 5, 9, 11],  # Características principais
            "Ciência da Computação": [2, 3, 5, 10, 11],
            "Engenharia Civil": [3, 6, 1, 4, 7],
            "Engenharia Elétrica": [3, 4, 7, 5, 10],
            "Matemática": [1, 5, 9, 3, 8]
        },
        "Humanas": {
            "Direito": [1, 3, 11, 4, 10],
            "Ciências da Saúde (Medicina, Enfermagem, Psicologia)": [2, 5, 7, 11, 6],
            "História": [0, 6, 7, 9, 10],
            "Marketing": [1, 3, 4, 8, 10],
            "Pedagogia": [0, 2, 5, 8, 9]
        }
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
            # --- PRIMEIRO GRÁFICO (Agrupamento Humanas/Exatas) ---
            fig1, ax1 = plt.subplots(figsize=(10, 6))  # Tamanho aumentado
            
            # Gerar pontos aleatórios para cada grupo
            np.random.seed(42)
            
            # Pontos para Humanas
            humanas_x = np.random.normal(0, 0.15, 20)
            humanas_y = np.random.normal(0, 0.15, 20)
            
            # Pontos para Exatas
            exatas_x = np.random.normal(1, 0.15, 20)
            exatas_y = np.random.normal(0, 0.15, 20)
            
            # Plotar grupos
            ax1.scatter(humanas_x, humanas_y, color='blue', alpha=0.6, label='Perfis de Humanas', s=80)
            ax1.scatter(exatas_x, exatas_y, color='green', alpha=0.6, label='Perfis de Exatas', s=80)
            
            # Plotar usuário
            user_x = 0 if st.session_state.perfil == "Humanas" else 1
            user_y = 0.3  # Posicionado acima dos outros pontos
            ax1.scatter(user_x, user_y, s=200, marker="*", 
                       color='red', label="Você", edgecolor='black')
            
            ax1.set_title("Seu Agrupamento na Primeira Etapa", pad=20)
            ax1.set_xlim(-0.5, 1.5)
            ax1.set_ylim(-0.5, 0.5)
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(["Humanas", "Exatas"])
            ax1.set_yticks([])
            ax1.legend(bbox_to_anchor=(1.05, 1))
            ax1.grid(True, linestyle="--", alpha=0.3)
            
            st.pyplot(fig1)
            
            # --- SEGUNDO GRÁFICO (Cursos específicos) ---
            # Criar vetor do usuário (1 para características selecionadas)
            vetor_usuario = np.array([1 if carac in selecoes else 0 for carac in caracteristicas])
            
            # Criar dados de treino baseados nos cursos
            dados_treino = []
            rotulos = []
            
            for curso, indices in cursos_map.items():
                # Criar 5 exemplos por curso com variações
                for _ in range(5):
                    vetor = np.zeros(len(caracteristicas))
                    # Ativar características principais
                    for idx in indices:
                        vetor[idx] = 1
                    # Adicionar variações aleatórias
                    vetor += np.random.normal(0, 0.2, len(vetor))
                    dados_treino.append(vetor)
                    rotulos.append(curso)
            
            dados_treino = np.array(dados_treino)
            
            # Clusterização com K-Means
            kmeans = KMeans(n_clusters=len(cursos_map), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(dados_treino)
            
            # Prever cluster do usuário
            cluster_usuario = kmeans.predict(vetor_usuario.reshape(1, -1))[0]
            
            # Encontrar curso mais frequente no cluster do usuário
            cursos_no_cluster = [rotulos[i] for i, c in enumerate(clusters) if c == cluster_usuario]
            from collections import Counter
            curso_ideal = Counter(cursos_no_cluster).most_common(1)[0][0]
            
            # Redução para 2D com PCA
            pca = PCA(n_components=2)
            dados_2d = pca.fit_transform(dados_treino)
            usuario_2d = pca.transform(vetor_usuario.reshape(1, -1))
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))  # Tamanho igual ao primeiro gráfico
            cores = plt.cm.get_cmap('tab10', len(cursos_map))
            
            # Mapeamento de curso para cor
            curso_para_indice = {curso: i for i, curso in enumerate(cursos_map.keys())}
            
            # Plotar pontos dos cursos (evitando labels duplicadas)
            handles = []
            labels = []
            for i, (x, y) in enumerate(dados_2d):
                curso = rotulos[i]
                color = cores(curso_para_indice[curso])
                if curso not in labels:
                    handle = ax2.scatter(x, y, color=color, label=curso, s=100, alpha=0.7)
                    handles.append(handle)
                    labels.append(curso)
                else:
                    ax2.scatter(x, y, color=color, s=100, alpha=0.7)
            
            # Calcular a posição ideal da estrela (centro do cluster do curso ideal)
            pontos_curso_ideal = dados_2d[[i for i, curso in enumerate(rotulos) if curso == curso_ideal]]
            centroide_curso = np.mean(pontos_curso_ideal, axis=0)
            
            # Plotar usuário no centroide do curso ideal, com pequeno deslocamento para visualização
            ax2.scatter(centroide_curso[0], centroide_curso[1] + 0.1, 
                      color=cores(curso_para_indice[curso_ideal]),
                      marker="*", s=300, edgecolor="black", label="Você")
            
            ax2.set_title("Sua Proximidade com os Cursos", pad=20)
            ax2.set_xlabel("Componente Principal 1")
            ax2.set_ylabel("Componente Principal 2")
            ax2.legend(handles=handles + [plt.Line2D([0], [0], marker='*', color='w', 
                                                    markerfacecolor='black', markersize=15, 
                                                    label='Você')],
                     bbox_to_anchor=(1.05, 1))
            ax2.grid(True, linestyle="--", alpha=0.3)
            
            st.pyplot(fig2)
            
            st.balloons()
            
            emoji_curso = {
                "Estatística": "📊", "Ciência da Computação": "💻",
                "Engenharia Civil": "🏗️", "Engenharia Elétrica": "⚡",
                "Matemática": "🧮", "Direito": "⚖️", 
                "Ciências da Saúde (Medicina, Enfermagem, Psicologia)": "🏥",
                "História": "🏛️", "Marketing": "📢", "Pedagogia": "📚"
            }.get(curso_ideal, "🎓")
            
            st.success(f"""
            **Resultado Final:**
            
            🎯 **Você tem perfil de {st.session_state.perfil}** e se encaixa melhor em:
            {emoji_curso} **{curso_ideal}**
            
            **Características que mais combinam:**
            """)
            
            # Listar características relevantes
            caracs_relevantes = [
                carac for i, carac in enumerate(caracteristicas) 
                if (i in cursos_map[curso_ideal] and st.session_state.segunda_etapa_respostas[i])
            ]
            
            for carac in caracs_relevantes[:5]:
                st.write(f"- {carac}")
            
            st.divider()
            st.markdown("""
            **📊 Como funciona a análise de perfil?**
            
            A técnica estatística conhecida como K-Means é amplamente utilizada em aplicativos de redes sociais como Instagram e TikTok. Já reparou que, ao criar uma conta no TikTok, ele pergunta que tipo de vídeos você gosta? Isso é parte de um processo de agrupamento, no qual o algoritmo tenta te colocar em um grupo com pessoas que têm preferências parecidas com as suas. Assim, ele identifica os estilos de vídeos que mais combinam com o seu perfil, com o objetivo de te manter engajado no aplicativo pelo maior tempo possível. Essa técnica também é usada para exibir anúncios que têm mais chance de agradar você. Entendeu agora por que às vezes aparece aquele anúncio exatamente sobre o que você estava pensando? Pois é... a estatística estava agindo o tempo todo — e você nem percebeu!
            """)

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()      
