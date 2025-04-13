import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("🔍 Descubra seu Perfil Acadêmico")
st.markdown("Responda algumas perguntas e veja em qual grupo você se encaixa!")

# --- Função para gerar dados simulados ---
def generate_simulated_data(shape, num_samples):
    return np.random.randint(0, 2, size=(num_samples, shape))

# --- Parte 1: Perguntas para identificar Humanas ou Exatas ---
st.subheader("Etapa 1: Seu estilo de pensamento")

start_button = st.button("Responder às perguntas")

if start_button:
    p1 = st.radio("Você prefere escrever ou resolver problemas lógicos?", ['Escrever', 'Resolver problemas'])
    p2 = st.radio("Você se interessa mais por leitura ou matemática?", ['Leitura', 'Matemática'])
    p3 = st.radio("Você prefere discutir ideias ou construir coisas?", ['Discutir ideias', 'Construir coisas'])
    p4 = st.radio("Você gosta mais de interpretar textos ou fazer cálculos?", ['Interpretar textos', 'Fazer cálculos'])

    respostas_1 = np.array([
        1 if p1 == 'Escrever' else 0,
        1 if p2 == 'Leitura' else 0,
        1 if p3 == 'Discutir ideias' else 0,
        1 if p4 == 'Interpretar textos' else 0
    ])

    # Gerando dados simulados para a fase 1
    X1 = generate_simulated_data(4, 100)
    X1 = np.vstack([X1, respostas_1])

    # Aplicando KMeans para separar em 2 grupos: Humanas vs Exatas
    kmeans1 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels1 = kmeans1.fit_predict(X1)
    user_group = labels1[-1]

    perfil = "Humanas" if user_group == 0 else "Exatas"
    st.success(f"Você tem mais afinidade com a área de **{perfil}**!")

    # --- Parte 2: Perguntas específicas por área ---
    st.subheader("Etapa 2: Qual curso combina com você?")

    if perfil == "Humanas":
        h1 = st.radio("Você gostaria de ensinar em escolas ou universidades?", ['Sim', 'Não'])
        h2 = st.radio("Você se interessa por leis, justiça ou debate?", ['Sim', 'Não'])
        h3 = st.radio("Você gosta de se comunicar, gravar vídeos ou escrever publicamente?", ['Sim', 'Não'])
        h4 = st.radio("Você gostaria de cuidar da saúde das pessoas?", ['Sim', 'Não'])
        h5 = st.radio("Você se vê atuando em áreas sociais ou comunitárias?", ['Sim', 'Não'])
        h6 = st.radio("Você gosta de ler sobre comportamento humano e sociedade?", ['Sim', 'Não'])

        respostas_2 = np.array([
            1 if h1 == 'Sim' else 0,
            1 if h2 == 'Sim' else 0,
            1 if h3 == 'Sim' else 0,
            1 if h4 == 'Sim' else 0,
            1 if h5 == 'Sim' else 0,
            1 if h6 == 'Sim' else 0
        ])

        # Gerando dados simulados para a fase 2 de Humanas
        X2 = generate_simulated_data(6, 100)
        X2 = np.vstack([X2, respostas_2])

        # Aplicando KMeans para separar os cursos da área de Humanas
        kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X2)
        curso_idx = labels2[-1]
        cursos = ['Professor(a)', 'Direito', 'Comunicação', 'Área Médica']
        curso_final = cursos[curso_idx]

    else:
        e1 = st.radio("Você gosta de resolver problemas matemáticos?", ['Sim', 'Não'])
        e2 = st.radio("Você se interessa por computadores e tecnologia?", ['Sim', 'Não'])
        e3 = st.radio("Você tem curiosidade sobre como o universo funciona?", ['Sim', 'Não'])
        e4 = st.radio("Você prefere lidar com dados e estatísticas do que com pessoas?", ['Sim', 'Não'])
        e5 = st.radio("Você gosta de construir, projetar ou inventar coisas?", ['Sim', 'Não'])
        e6 = st.radio("Você tem interesse em lógica, quebra-cabeças ou jogos de estratégia?", ['Sim', 'Não'])

        respostas_2 = np.array([
            1 if e1 == 'Sim' else 0,
            1 if e2 == 'Sim' else 0,
            1 if e3 == 'Sim' else 0,
            1 if e4 == 'Sim' else 0,
            1 if e5 == 'Sim' else 0,
            1 if e6 == 'Sim' else 0
        ])

        # Gerando dados simulados para a fase 2 de Exatas
        X2 = generate_simulated_data(6, 100)
        X2 = np.vstack([X2, respostas_2])

        # Aplicando KMeans para separar os cursos da área de Exatas
        kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X2)
        curso_idx = labels2[-1]
        cursos = ['Estatística/Matemática', 'Engenharia', 'Física', 'Ciência da Computação']
        curso_final = cursos[curso_idx]

    # Botão final para exibir resultado
    if st.button("Ver meu resultado final e gráficos"):
        st.info(f"Você tem mais perfil para o curso de **{curso_final}**!")

        # --- Visualização ---
        st.subheader("Visualização dos Agrupamentos")
        pca1 = PCA(n_components=2)
        pca_data1 = pca1.fit_transform(X1)
        pca2 = PCA(n_components=2)
        pca_data2 = pca2.fit_transform(X2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        scatter1 = ax1.scatter(pca_data1[:, 0], pca_data1[:, 1], c=labels1, cmap='coolwarm', alpha=0.6, edgecolor='k', marker='o')
        ax1.set_title("Fase 1: Humanas vs Exatas")
        ax1.set_xlabel("Eixo X: Preferências cognitivas")
        ax1.set_ylabel("Eixo Y: Estilo de raciocínio")

        scatter2 = ax2.scatter(pca_data2[:, 0], pca_data2[:, 1], c=labels2, cmap='tab10', alpha=0.6, edgecolor='k', marker='o')
        ax2.set_title("Fase 2: Perfil dentro da área escolhida")
        ax2.set_xlabel("Eixo X: Interesse específico")
        ax2.set_ylabel("Eixo Y: Afinidade comportamental")

        st.pyplot(fig)

        st.caption("\n\nCada ponto representa uma pessoa. Os agrupamentos são formados com base nas respostas dadas. Com o tempo, o sistema aprende com os usuários!")
