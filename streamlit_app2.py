import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
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
            st.session_state.respostas = respostas
            st.session_state.etapa = 2
            st.session_state.segunda_etapa_respostas = [False] * 12  # 12 características
            st.rerun()

# --- SEGUNDA ETAPA ---
elif st.session_state.etapa == 2:
    # Classificação inicial (Humanas/Exatas)
    perfil = "Humanas" if sum(st.session_state.respostas) < 5 else "Exatas"
    st.session_state.perfil = perfil

    st.success(f"Perfil principal: **{perfil}**")
    st.divider()
    st.subheader("📌 **Parte 2/2:** Selecione as características que mais combinam com você")
    
    # Características por área (12 opções) - Definições mais específicas
    caracteristicas = {
        "Exatas": [
            "📊 Criar modelos estatísticos complexos",
            "🧮 Desenvolver teorias matemáticas abstratas",
            "⚡ Projetar circuitos elétricos complexos",
            "🏗️ Calcular estruturas de concreto armado",
            "💻 Desenvolver algoritmos de IA",
            "📈 Analisar tendências de mercado",
            "🔢 Resolver equações diferenciais",
            "📐 Projetar sistemas mecânicos",
            "🌐 Otimizar redes de computadores",
            "🧪 Simular experimentos físicos",
            "🤖 Programar robôs autônomos",
            "📉 Visualizar dados multivariados"
        ],
        "Humanas": [
            "⚖️ Argumentar casos jurídicos complexos",
            "📜 Analisar fontes históricas primárias",
            "📖 Escrever crítica literária",
            "🧠 Aplicar testes psicológicos",
            "🗣️ Mediar conflitos organizacionais",
            "🎨 Criar exposições artísticas",
            "🌍 Estudar antropologia cultural",
            "✍️ Produzir textos acadêmicos",
            "🏛️ Interpretar artefatos arqueológicos",
            "👥 Conduzir terapia de grupo",
            "💬 Desenvolver discursos persuasivos",
            "📝 Elaborar pareceres técnicos"
        ]
    }[perfil]

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
                    "Estatística": [0, 5, 11],
                    "Matemática": [1, 6],
                    "Engenharia Elétrica": [2],
                    "Engenharia Civil": [3],
                    "Ciência da Computação": [4, 8],
                    "Engenharia Mecânica": [7],
                    "Engenharia de Controle": [10]
                },
                "Humanas": {
                    "Direito": [0, 11],
                    "História": [1, 8],
                    "Letras": [2, 7],
                    "Psicologia": [3, 9],
                    "Artes": [5],
                    "Antropologia": [6],
                    "Comunicação": [10]
                }
            }[perfil]
            
            # Calcular pontuação para cada curso
            scores = {}
            for curso, idx_caracs in cursos_map.items():
                score = sum(1 for idx in idx_caracs 
                           if caracteristicas[idx] in selecoes)
                scores[curso] = score
            
            # Curso ideal é o com maior pontuação
            curso_ideal = max(scores.items(), key=lambda x: x[1])[0]
            
            # --- VISUALIZAÇÃO ---
            # Gerar pontos para os cursos (3 por curso)
            pontos_curso = {}
            for curso in cursos_map.keys():
                base_pos = list(cursos_map.keys()).index(curso)
                pontos_curso[curso] = np.column_stack([
                    np.random.normal(base_pos, 0.1, size=3),
                    np.random.normal(0, 0.1, size=3)
                ])
            
            # Posição do usuário (próxima ao curso ideal)
            user_pos = np.mean(pontos_curso[curso_ideal], axis=0) + np.array([0, 0.2])
            
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            cores = plt.cm.tab10.colors
            
            for i, (curso, pontos) in enumerate(pontos_curso.items()):
                ax.scatter(
                    pontos[:, 0], pontos[:, 1],
                    color=cores[i],
                    s=100,
                    label=f"{curso} ({scores[curso]})",
                    alpha=0.7
                )
            
            ax.scatter(
                user_pos[0], user_pos[1],
                color=cores[list(cursos_map.keys()).index(curso_ideal)],
                marker="*",
                s=300,
                edgecolor="black",
                label=f"Você → {curso_ideal}"
            )
            
            ax.set_title("Sua Proximidade com os Cursos", pad=20)
            ax.set_xticks(range(len(cursos_map)))
            ax.set_xticklabels(cursos_map.keys(), rotation=45)
            ax.set_yticks([])
            ax.legend(bbox_to_anchor=(1.05, 1))
            ax.grid(True, linestyle="--", alpha=0.3)
            
            st.pyplot(fig)
            
            # Resultado Final
            st.balloons()
            emoji_curso = {
                "Estatística": "📊", "Matemática": "🧮", 
                "Engenharia Elétrica": "⚡", "Engenharia Civil": "🏗️",
                "Ciência da Computação": "💻", "Engenharia Mecânica": "⚙️",
                "Engenharia de Controle": "🤖", "Direito": "⚖️",
                "História": "🏛️", "Letras": "📖", "Psicologia": "🧠",
                "Artes": "🎨", "Antropologia": "🌍", "Comunicação": "💬"
            }.get(curso_ideal, "🎓")
            
            st.success(f"""
            **Resultado Final:**
            
            🎯 **Você tem perfil de {perfil}** e se encaixa melhor em:
            {emoji_curso} **{curso_ideal}**
            
            **Características selecionadas que mais combinam:**
            """)
            
            # Corrigindo a lista de características correspondentes
            caracs_correspondentes = [
                carac for carac in selecoes 
                if any(carac == caracteristicas[idx] for idx in cursos_map[curso_ideal])
            ]
            
            for carac in caracs_correspondentes:
                st.write(f"- {carac}")

    if st.button("↩️ Voltar para a Parte 1"):
        st.session_state.etapa = 1
        st.rerun()   
