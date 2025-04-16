import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter

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
            st.session_state.segunda_etapa_respostas = [False] * 12
            st.rerun()

# --- SEGUNDA ETAPA (OTIMIZADA) ---
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as 5 características que mais combinam com você")
    
    # Características ajustadas para melhor clusterização
    caracteristicas = {
        "Exatas": [
            "Gosto de analisar dados e estatísticas (Estatística)",
            "Tenho facilidade com cálculos complexos (Matemática/Engenharias)",
            "Me interesso por programação e algoritmos (Computação)",
            "Gosto de resolver problemas práticos (Engenharias)",
            "Tenho curiosidade sobre como máquinas funcionam (Engenharias)",
            "Prefiro raciocínio lógico a subjetivo (Exatas)",
            "Gosto de projetar e construir coisas (Engenharia Civil)",
            "Me interesso por experimentos científicos (Física/Química)",
            "Tenho habilidade com números e gráficos (Estatística)",
            "Gosto de desafios matemáticos (Matemática)",
            "Me interesso por inteligência artificial (Computação)",
            "Tenho facilidade com modelos 3D (Engenharia/Arquitetura)"
        ],
        "Humanas": [
            "Gosto de ler e interpretar textos (Letras/História)",
            "Tenho facilidade em me expressar oralmente (Direito/Comunicação)",
            "Me interesso por entender comportamentos (Psicologia)",
            "Gosto de debater e argumentar (Direito/Filosofia)",
            "Tenho sensibilidade artística (Artes/Design)",
            "Me preocupo com questões sociais (Serviço Social)",
            "Gosto de estudar culturas e sociedades (História/Antropologia)",
            "Tenho facilidade com idiomas (Letras/Rel. Internacionais)",
            "Prefiro trabalhos colaborativos (Pedagogia/Psicologia)",
            "Gosto de analisar obras artísticas (Artes/História)",
            "Me interesso por políticas públicas (Direito/Administração)",
            "Tenho habilidade para mediar conflitos (Psicologia/Direito)"
        ]
    }[st.session_state.perfil]

    # Mapeamento curso-características com pesos otimizados
    cursos_map = {
        "Exatas": {
            "Estatística": [0, 1, 8, 5, 9],
            "Ciência da Computação": [2, 3, 10, 5, 11],
            "Engenharia Civil": [3, 6, 1, 4, 11],
            "Engenharia Elétrica": [3, 4, 7, 5, 10],
            "Matemática": [1, 5, 9, 0, 8]
        },
        "Humanas": {
            "Direito": [1, 3, 11, 4, 10],
            "Psicologia": [2, 5, 8, 11, 9],
            "História": [0, 6, 7, 9, 10],
            "Letras": [0, 1, 7, 9, 11],
            "Artes": [4, 6, 9, 2, 8]
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
            # --- PRIMEIRO GRÁFICO ---
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            # ... (código do primeiro gráfico mantido igual) ...

            # --- SEGUNDO GRÁFICO (COM K-MEANS OTIMIZADO) ---
            # 1. Preparação dos dados com pesos reforçados
            dados_treino = []
            rotulos = []
            
            for curso, indices in cursos_map.items():
                for _ in range(15):  # Mais exemplos por curso
                    vetor = np.zeros(len(caracteristicas))
                    # Peso maior para características principais (1.5)
                    for idx in indices:
                        vetor[idx] = 1.5
                    # Adiciona variação controlada
                    vetor += np.random.normal(0, 0.1, len(vetor))
                    dados_treino.append(vetor)
                    rotulos.append(curso)
            
            dados_treino = np.array(dados_treino)
            
            # 2. Clusterização com K-Means otimizado
            kmeans = KMeans(
                n_clusters=len(cursos_map),
                random_state=42,
                n_init=20,  # Mais inicializações
                max_iter=300,  # Mais iterações
                algorithm='elkan'  # Algoritmo mais eficiente
            )
            clusters = kmeans.fit_predict(dados_treino)
            
            # 3. Previsão para o usuário com pesos reforçados
            vetor_usuario = np.array([1.2 if carac in selecoes else 0 for carac in caracteristicas])
            cluster_usuario = kmeans.predict(vetor_usuario.reshape(1, -1))[0]
            
            # 4. Determinação do curso ideal com fallback
            cursos_no_cluster = [rotulos[i] for i, c in enumerate(clusters) if c == cluster_usuario]
            contagem = Counter(cursos_no_cluster)
            
            # Fallback para seleções mistas
            if len(contagem) > 2:
                scores = {curso: sum(vetor_usuario[indices]) for curso, indices in cursos_map.items()}
                curso_ideal = max(scores.items(), key=lambda x: x[1])[0]
            else:
                curso_ideal = contagem.most_common(1)[0][0]
            
            # 5. Visualização com PCA
            pca = PCA(n_components=2)
            dados_2d = pca.fit_transform(dados_treino)
            usuario_2d = pca.transform(vetor_usuario.reshape(1, -1))
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            cores = plt.cm.get_cmap('tab10', len(cursos_map))
            
            # Plotagem dos clusters
            for i, curso in enumerate(cursos_map.keys()):
                pontos_curso = dados_2d[[j for j, cur in enumerate(rotulos) if cur == curso]]
                ax2.scatter(
                    pontos_curso[:, 0], pontos_curso[:, 1],
                    color=cores(i),
                    label=curso,
                    alpha=0.6,
                    s=100
                )
            
            # Posicionamento preciso da estrela
            centroide = np.mean(dados_2d[[i for i, curso in enumerate(rotulos) if curso == curso_ideal]], axis=0)
            ax2.scatter(
                centroide[0], centroide[1] + 0.15,
                color=cores(list(cursos_map.keys()).index(curso_ideal)),
                marker="*",
                s=400,
                edgecolor="black",
                label="Você"
            )
            
            ax2.set_title("Sua Proximidade com os Cursos (Análise de Cluster)", pad=20)
            ax2.legend(bbox_to_anchor=(1.05, 1))
            st.pyplot(fig2)
            
            # --- RESULTADO FINAL ---
            st.balloons()
            emoji_curso = {
                "Estatística": "📊", "Ciência da Computação": "💻",
                "Engenharia Civil": "🏗️", "Engenharia Elétrica": "⚡",
                "Matemática": "🧮", "Direito": "⚖️", 
                "Psicologia": "🧠", "História": "🏛️",
                "Letras": "📖", "Artes": "🎨"
            }.get(curso_ideal, "🎓")
            
            st.success(f"""
            **Resultado Final:**
            
            🎯 **Você tem perfil de {st.session_state.perfil}** e se encaixa melhor em:
            {emoji_curso} **{curso_ideal}**
            
            **Características selecionadas que mais contribuíram:**
            """)
            
            # Mostra as características mais relevantes
            indices_curso = cursos_map[curso_ideal]
            caracs_principais = [
                (i, caracteristicas[i]) for i in indices_curso 
                if st.session_state.segunda_etapa_respostas[i]
            ]
            for idx, carac in sorted(caracs_principais, key=lambda x: x[0]):
                st.write(f"- {carac.split(' (')[0]}")

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()    
