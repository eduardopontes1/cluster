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

# --- PRIMEIRA ETAPA: CLASSIFICAÇÃO HUMANAS/EXATAS ---
if st.session_state.etapa == 1:
    st.write("**Parte 1/2:** Marque os conteúdos com que você mais se identifica:")
    
    itens = [
        {"texto": "Escrever poemas ou crônicas", "valor": 0},
        {"texto": "Resolver desafios de programação", "valor": 1},
        {"texto": "Debater sobre filosofia", "valor": 0},
        {"texto": "Projetar experimentos científicos", "valor": 1}
    ]
    
    respostas = [0] * len(itens)
    for i, item in enumerate(itens):
        if st.checkbox(item["texto"], key=f"item_{i}"):
            respostas[i] = 1

    if st.button("🔎 Avançar para a Parte 2"):
        if sum(respostas) == 0:
            st.warning("Selecione pelo menos um conteúdo!")
        else:
            # Simulação do K-means (dados simplificados)
            X_novo = np.array(respostas).reshape(1, -1)
            X_treino = np.array([[1,0,1,0], [0,1,0,1], [1,0,0,0], [0,1,1,0]])
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_treino)
            
            st.session_state.perfil = "Humanas" if kmeans.predict(X_novo)[0] == 0 else "Exatas"
            st.session_state.etapa = 2
            st.rerun()

# --- SEGUNDA ETAPA: IDENTIFICAÇÃO DO CURSO IDEAL ---
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Identifique seu curso ideal")
    
    # Dicionário de cursos e características
    cursos = {
        "Exatas": {
            "Estatística": ["📊 Analisar dados", "🧮 Resolver problemas matemáticos", "📈 Trabalhar com probabilidades"],
            "Engenharia Elétrica": ["⚡ Projetar circuitos", "🔌 Trabalhar com eletrônica", "💡 Solucionar problemas físicos"],
            "Matemática": ["∞ Trabalhar com abstrações", "➗ Desenvolver teoremas", "✏️ Resolver equações complexas"]
        },
        "Humanas": {
            "Direito": ["⚖️ Argumentar juridicamente", "📜 Interpretar leis", "🗣️ Debater casos"],
            "Psicologia": ["🧠 Analisar comportamentos", "👂 Ouvir ativamente", "💬 Interpretar emoções"],
            "Letras": ["📖 Escrever textos criativos", "🔍 Analisar obras literárias", "👩‍🏫 Ensinar gramática"]
        }
    }[st.session_state.perfil]

    # Widgets para seleção de características
    st.write("**Marque as características que mais combinam com você:**")
    caracteristicas_selecionadas = []
    
    cols = st.columns(3)
    for i, (curso, caracs) in enumerate(cursos.items()):
        with cols[i % 3]:
            st.markdown(f"**{curso}**")
            for carac in caracs:
                if st.checkbox(carac, key=f"carac_{curso}_{carac}"):
                    caracteristicas_selecionadas.append(carac)

    if st.button("🎯 Descobrir meu curso ideal"):
        if len(caracteristicas_selecionadas) < 2:
            st.warning("Selecione pelo menos 2 características!")
        else:
            # Preparação dos dados para clustering
            X_curso = np.array([
                [1, 1, 1, 0, 0, 0, 0, 0, 0],  # Estatística
                [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Eng. Elétrica
                [0, 0, 0, 0, 0, 0, 1, 1, 1]   # Matemática
            ] if st.session_state.perfil == "Exatas" else [
                [1, 1, 1, 0, 0, 0, 0, 0, 0],  # Direito
                [0, 0, 0, 1, 1, 1, 0, 0, 0],  # Psicologia
                [0, 0, 0, 0, 0, 0, 1, 1, 1]   # Letras
            ])
            
            # Simulação da resposta do usuário (baseado nas seleções)
            X_usuario = np.array([1 if any(c in carac for carac in caracteristicas_selecionadas) else 0 
                                for c in range(9)]).reshape(1, -1)
            
            # K-means para cursos
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_curso)
            cluster_usuario = kmeans.predict(X_usuario)[0]
            curso_ideal = list(cursos.keys())[cluster_usuario]
            
            # Visualização com PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(np.vstack((X_curso, X_usuario)))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            cores = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
            formas = ["o", "s", "D"]
            
            for i in range(len(X_curso)):
                ax.scatter(
                    X_2d[i, 0], X_2d[i, 1],
                    marker=formas[i],
                    color=cores[i],
                    s=200,
                    label=list(cursos.keys())[i]
                )
            
            ax.scatter(
                X_2d[-1, 0], X_2d[-1, 1],
                marker="*",
                color="gold",
                s=400,
                edgecolor="black",
                label="Você"
            )
            
            ax.set_title("Agrupamento de Cursos por Características", pad=20)
            ax.legend(bbox_to_anchor=(1.25, 1))
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)
            
            st.balloons()
            st.success(f"**Curso ideal:** {curso_ideal}")
            
            # Destaque para Estatística
            if curso_ideal == "Estatística":
                st.markdown("""
                🎉 **Você tem o perfil perfeito para Estatística!**  
                📌 Áreas de atuação:  
                - Análise de dados  
                - Pesquisa científica  
                - Inteligência artificial  
                """)

    # Botão para voltar à primeira etapa
    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()
