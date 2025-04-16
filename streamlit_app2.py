import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Configuração da página
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico ")

# Variáveis de sessão para controle de estado
if 'etapa' not in st.session_state:
    st.session_state.etapa = 1
    st.session_state.perfil = None
    st.session_state.respostas_curso = {}

# --- PRIMEIRA ETAPA: CLASSIFICAÇÃO HUMANAS/EXATAS (COM MAIS ITENS) ---
if st.session_state.etapa == 1:
    st.write("**Parte 1/2:** Marque os conteúdos com que você mais se identifica:")
    
    itens = [
        {"texto": "Escrever poemas ou crônicas", "valor": 0},
        {"texto": "Resolver desafios de programação", "valor": 1},
        {"texto": "Debater sobre filosofia ou sociologia", "valor": 0},
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
        if sum(respostas) < 3:  # Pelo menos 3 seleções
            st.warning("Selecione pelo menos 3 conteúdos!")
        else:
            # Simulação do K-means (dados simplificados)
            X_novo = np.array(respostas).reshape(1, -1)
            grupo_humanas = np.array([[1,0,1,0,1,0,1,0,1,0], [0,0,1,0,1,0,0,0,1,0]])
            grupo_exatas = np.array([[0,1,0,1,0,1,0,1,0,1], [1,1,0,1,0,1,0,0,0,1]])
            X_treino = np.vstack((grupo_humanas, grupo_exatas))
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_treino)
            st.session_state.perfil = "Humanas" if kmeans.predict(X_novo)[0] == 0 else "Exatas"
            st.session_state.etapa = 2
            st.rerun()

# --- SEGUNDA ETAPA: IDENTIFICAÇÃO DO CURSO IDEAL (CORRIGIDA) ---
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Identifique seu curso ideal")
    
    # Dicionário de cursos e características
    cursos = {
        "Exatas": {
            "Estatística": ["📊 Analisar dados", "🧮 Resolver problemas matemáticos", "📈 Trabalhar com probabilidades"],
            "Engenharia Elétrica": ["⚡ Projetar circuitos", "🔌 Trabalhar com eletrônica", "💡 Solucionar problemas físicos"],
            "Matemática": ["∞ Trabalhar com abstrações", "➗ Desenvolver teoremas", "✏️ Resolver equações complexas"],
            "Ciência da Computação": ["💻 Programar algoritmos", "🤖 Desenvolver inteligência artificial", "🔢 Trabalhar com estruturas de dados"]
        },
        "Humanas": {
            "Direito": ["⚖️ Argumentar juridicamente", "📜 Interpretar leis", "🗣️ Debater casos"],
            "Psicologia": ["🧠 Analisar comportamentos", "👂 Ouvir ativamente", "💬 Interpretar emoções"],
            "Letras": ["📖 Escrever textos criativos", "🔍 Analisar obras literárias", "👩‍🏫 Ensinar gramática"],
            "História": ["🏛️ Estudar civilizações antigas", "📜 Analisar documentos históricos", "🌍 Compreender contextos culturais"]
        }
    }[st.session_state.perfil]

    # Widgets para seleção de características
    st.write("**Marque as características que mais combinam com você (selecione 3 a 5):**")
    caracteristicas_selecionadas = []
    
    # Organiza em colunas
    cols = st.columns(2)
    for i, (curso, caracs) in enumerate(cursos.items()):
        with cols[i % 2]:
            st.markdown(f"**{curso}**")
            for carac in caracs:
                if st.checkbox(carac, key=f"carac_{curso}_{carac}"):
                    caracteristicas_selecionadas.append((curso, carac))

    if st.button("🎯 Descobrir meu curso ideal"):
        if len(caracteristicas_selecionadas) < 3:
            st.warning("Selecione pelo menos 3 características!")
        else:
            # Mapeamento de características para vetores numéricos
            all_caracs = [carac for curso in cursos.values() for carac in curso]
            carac_to_idx = {carac: idx for idx, carac in enumerate(all_caracs)}
            
            # Vetor do usuário (one-hot encoding)
            X_usuario = np.zeros(len(all_caracs))
            for _, carac in caracteristicas_selecionadas:
                X_usuario[carac_to_idx[carac]] = 1
            
            # Vetores de referência para cada curso (one-hot)
            X_cursos = []
            for curso, caracs in cursos.items():
                vec = np.zeros(len(all_caracs))
                for carac in caracs:
                    vec[carac_to_idx[carac]] = 1
                X_cursos.append(vec)
            X_cursos = np.array(X_cursos)
            
            # K-means para cursos
            kmeans = KMeans(n_clusters=len(cursos), random_state=42, n_init=10).fit(X_cursos)
            
            # Determinar curso ideal (cluster com maior sobreposição)
            cluster_usuario = kmeans.predict(X_usuario.reshape(1, -1))[0]
            curso_ideal = list(cursos.keys())[cluster_usuario]
            
            # Visualização com PCA
            pca = PCA(n_components=2)
            X_combined = np.vstack((X_cursos, X_usuario))
            X_2d = pca.fit_transform(X_combined)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            cores = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#9B59B6"]  # 4 cores distintas
            formas = ["o", "s", "D", "^"]  # Marcadores diferentes
            
            for i in range(len(X_cursos)):
                ax.scatter(
                    X_2d[i, 0], X_2d[i, 1],
                    marker=formas[i],
                    color=cores[i],
                    s=200,
                    label=list(cursos.keys())[i]
                )
            
            # Posição do usuário (ajustada para ficar próximo ao cluster correto)
            ax.scatter(
                X_2d[-1, 0] + np.random.normal(0, 0.1),  # Pequeno ruído para visualização
                X_2d[-1, 1] + np.random.normal(0, 0.1),
                marker="*",
                color="gold",
                s=400,
                edgecolor="black",
                label="Você"
            )
            
            ax.set_title("Agrupamento de Cursos por Características", pad=20)
            ax.legend(bbox_to_anchor=(1.3, 1))
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)
            
            st.balloons()
            st.success(f"**Curso ideal:** {curso_ideal}")
            
            # Destaque especial para Estatística
            if curso_ideal == "Estatística":
                st.markdown("""
                🎉 **Você tem o perfil perfeito para Estatística!**  
                📌 Áreas de atuação:  
                - Ciência de Dados  
                - Pesquisa Científica  
                - Inteligência Artificial  
                - Mercado Financeiro  
                """)

    # Botão para voltar à primeira etapa
    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()
