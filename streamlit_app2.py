import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("🔍 Descubra seu perfil Acadêmico")
st.markdown("""
            **📊 Como funciona a análise de perfil?**
            
            A técnica estatística conhecida como K-Means é amplamente utilizada em aplicativos de redes sociais como Instagram e TikTok. Já reparou que, 
            ao criar uma conta no TikTok, ele pergunta que tipo de vídeos você gosta? Isso é parte de um processo de agrupamento, no qual o algoritmo 
            tenta te colocar em um grupo com pessoas que têm preferências parecidas com as suas. Assim, ele identifica os estilos de vídeos que mais 
            combinam com o seu perfil, com o objetivo de te manter engajado no aplicativo pelo maior tempo possível. Essa técnica também é usada para 
            exibir anúncios que têm mais chance de agradar você.
            """)
if 'etapa' not in st.session_state:
    st.session_state.etapa = 1
    st.session_state.perfil = None
    st.session_state.respostas = None
    st.session_state.segunda_etapa_respostas = None
if st.session_state.etapa == 1:
    st.write("Marque os conteúdos com que você mais se identifica:")
    itens = [
        {"texto": "Escrever/ler poemas ou crônicas", "valor": 0},
        {"texto": "Resolver desafios de programação", "valor": 1},
        {"texto": "Debater sobre filosofia/sociologia", "valor": 0},
        {"texto": "Projetar experimentos científicos", "valor": 1},
        {"texto": "Me exercitar", "valor": 0},
        {"texto": "Desenvolver fórmulas matemáticas", "valor": 1},
        {"texto": "Ler sobre política internacional", "valor": 0},
        {"texto": "Estudar novas tecnologias", "valor": 1},
        {"texto": "Cuidar das pessoas", "valor": 0},
        {"texto": "Trabalhar com cálculos complexos", "valor": 1}
    ]
    respostas = [0] * len(itens)
    for i, item in enumerate(itens):
        if st.checkbox(item["texto"], key=f"item_{i}"):
            respostas[i] = 1
    if st.button("🔎 Avançar"):
        if sum(respostas) < 3:
            st.warning("Selecione pelo menos 3 conteúdos!")
        else:
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
elif st.session_state.etapa == 2:
    st.success(f"Perfil principal: **{st.session_state.perfil}**")
    st.divider()
    st.subheader("📌Selecione as 5 características que mais combinam com você")
    caracteristicas = {
        "Exatas": [
            "Gosto de analisar dados e padrões",
            "Tenho facilidade com cálculos complexos",
            "Me interesso por programação e algoritmos",
            "Gosto de resolver problemas práticos",
            "Tenho curiosidade sobre como as coisas funcionam",
            "Prefiro raciocínio lógico a subjetivo",
            "Gosto de projetar e construir coisas",
            "Me interesso por experimentos científicos",
            "Tenho habilidade com números e gráficos",
            "Gosto de entender reações e transformações",
            "Me interesso por tecnologia avançada",
            "Gosto de resolver enigmas"
        ],
        "Humanas": [
            "Gosto de ler e interpretar textos",
            "Tenho facilidade em me expressar oralmente",
            "Me interesso por entender comportamentos",
            "Gosto de debater e argumentar",
            "Sou criativo",
            "Me preocupo com questões sociais",
            "Gosto de me exercitar",
            "Tenho facilidade com idiomas",
            "Prefiro trabalhos colaborativos",
            "Gosto de analisar",
            "Me interesso por questões políticas",
            "Tenho habilidade para mediar conflitos"
        ]
    }[st.session_state.perfil]
    cursos_map = {
        "Exatas": {
            "Estatística": [0, 1, 5, 8, 11],
            "Ciência da Computação": [2, 3, 5, 10, 11],
            "Engenharia Civil": [3, 6, 1, 4, 7],
            "Engenharia Elétrica": [3, 4, 7, 5, 10],
            "Química": [4, 7, 9, 1, 5]
        },
        "Humanas": {
            "Direito": [1, 3, 11, 4, 10],
            "Medicina/Psicologia/Odontologia": [2, 5, 8, 11, 9],
            "Educação Física": [1, 2, 8, 8, 9],
            "Letras": [0, 1, 7, 9, 11],
            "Marketing": [4, 0, 9, 2, 8]
        }
    }[st.session_state.perfil]
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
            # 1. Preparação dos dados com pesos reforçados
            dados_treino = []
            rotulos = []
            
            for curso, indices in cursos_map.items():
                for _ in range(15):  # Mais exemplos por curso
                    vetor = np.zeros(len(caracteristicas))
                    # Peso maior para características principais
                    for idx in indices:
                        vetor[idx] = 1.5
                    # Adiciona variação controlada
                    vetor += np.random.normal(0, 0.1, len(vetor))
                    dados_treino.append(vetor)
                    rotulos.append(curso)
            
            dados_treino = np.array(dados_treino)
            kmeans = KMeans(
                n_clusters=len(cursos_map),
                random_state=42,
                n_init=20,
                max_iter=300,
                algorithm='elkan'
            )
            clusters = kmeans.fit_predict(dados_treino)
            vetor_usuario = np.array([1.2 if carac in selecoes else 0 for carac in caracteristicas])
            cluster_usuario = kmeans.predict(vetor_usuario.reshape(1, -1))[0]
            cursos_no_cluster = [rotulos[i] for i, c in enumerate(clusters) if c == cluster_usuario]
            contagem = Counter(cursos_no_cluster)
            if len(contagem) > 2:
                scores = {curso: sum(vetor_usuario[indices]) for curso, indices in cursos_map.items()}
                curso_ideal = max(scores.items(), key=lambda x: x[1])[0]
            else:
                curso_ideal = contagem.most_common(1)[0][0]
            st.balloons()
            emoji_curso = {
                "Estatística": "📊", "Ciência da Computação": "💻",
                "Engenharia Civil": "🏗️", "Engenharia Elétrica": "⚡",
                "Química": "🧪", "Direito": "⚖️", 
                "Medicina/Psicologia/Odontologia": "🧠", "História": "🏛️",
                "Letras": "📖", "Marketing": "🎨"
            }.get(curso_ideal, "🎓")
            
            st.success(f"""
            **Resultado Final:**
            
            🎯 **Você tem perfil de {st.session_state.perfil}** e se encaixa melhor em:
            {emoji_curso} **{curso_ideal}**
            
            **Características selecionadas que mais contribuíram:**
            """)
            indices_curso = cursos_map[curso_ideal]
            caracs_principais = [
                (i, caracteristicas[i]) for i in indices_curso 
                if st.session_state.segunda_etapa_respostas[i]
            ]
            for idx, carac in sorted(caracs_principais, key=lambda x: x[0]):
                st.write(f"- {carac}")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            np.random.seed(42)
            humanas_x = np.random.normal(0, 0.15, 20)
            humanas_y = np.random.normal(0, 0.15, 20)
            exatas_x = np.random.normal(1, 0.15, 20)
            exatas_y = np.random.normal(0, 0.15, 20)
            ax1.scatter(humanas_x, humanas_y, color='blue', alpha=0.6, label='Perfis de Humanas', s=80)
            ax1.scatter(exatas_x, exatas_y, color='green', alpha=0.6, label='Perfis de Exatas', s=80)
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
            ax1.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
            ax1.grid(True, linestyle="--", alpha=0.3)
       
            st.pyplot(fig1)
            
            pca = PCA(n_components=2)
            dados_2d = pca.fit_transform(dados_treino)
            usuario_2d = pca.transform(vetor_usuario.reshape(1, -1))
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            cores = plt.cm.get_cmap('tab10', len(cursos_map))
            
            
            for i, curso in enumerate(cursos_map.keys()):
                pontos_curso = dados_2d[[j for j, cur in enumerate(rotulos) if cur == curso]]
                ax2.scatter(
                    pontos_curso[:, 0], pontos_curso[:, 1],
                    color=cores(i),
                    label=curso,
                    alpha=0.6,
                    s=100
                )
            
            
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
            ax2.legend(bbox_to_anchor=(1.05,1),loc='upper left', borderaxespad=0.)
            st.pyplot(fig2)

            st.divider()
            st.markdown(""" 
            Na internet, muitas vezes não entregamos nossos dados de forma direta, mas basta assistir certos tipos de vídeos por mais tempo ou clicar em determinados conteúdos 
            para que os algoritmos comecem a nos entender. Com base nesses padrões de comportamento, somos agrupados em perfis que se parecem com o nosso – tudo
            isso por meio de técnicas como o KMeans. Assim, fica fácil para as redes sociais nos mostrarem conteúdos que parecem feitos sob medida. Entendeu agora como elas acertam 
            tanto? Era a estatística agindo o tempo todo... e você nem percebeu.
            """)

    if st.button("↩️ Voltar"):
        st.session_state.etapa = 1
        st.rerun()
