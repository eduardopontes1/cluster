import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
st.set_page_config(page_title="Perfil Acadêmico", layout="centered")
st.title("Descubra seu perfil Acadêmico")
st.markdown("""
            **Como funciona a análise de perfil?**
            
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
        {"texto": "Me interesso por entender comportamentos humanos e emoções.", "valor": 0},
        {"texto": "Gosto de fazer cursos que me ensinem mais sobre tecnologia.", "valor": 1},
        {"texto": "Tenho interesse em ensinar, orientar ou mediar conflitos.", "valor": 0},
        {"texto": "Tenho interesse em seguir uma carreira que use habilidades tecnológicas.", "valor": 1},
        {"texto": "Me vejo liderando, organizando pessoas ou projetos com impacto social.", "valor": 0},
        {"texto": "Gosto de fazer cursos que me ensinem mais sobre ciência.", "valor": 1},
        {"texto": "Me interesso por expressar ideias de forma criativa ou estética.", "valor": 0},
        {"texto": "Tenho interesse em seguir uma carreira que use habilidades científicas.", "valor": 1},
        {"texto": "Gosto de ajudar pessoas a resolverem seus problemas.", "valor": 0},
        {"texto": "Tenho interesse em seguir uma carreira que use habilidades matemáticas.", "valor": 1}
    ]
    respostas = [0] * len(itens)
    for i, item in enumerate(itens):
        if st.checkbox(item["texto"], key=f"item_{i}"):
            respostas[i] = 1
    if st.button("Avançar"):
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
    st.subheader("Na escola, prefiro atividades...")
    caracteristicas = {
        "Exatas": [
            "...com uso de fórmulas e análise de gráficos!",
            "...que envolvem lógica, matemática e organização!",
            "...que envolvem programar ou criar sistemas digitais!",
            "...com tecnologias novas e inovadoras!",
            "...de laboratórios e experimentos científicos!",
            "...em que eu aplico conhecimentos matemáticos na prática!",
            "...que combinam teoria com prática técnica!",
            "...que simulam construção e planejamento de estruturas!",
            "...de simulação ou modelagem de sistemas reais!",
            "...voltadas para entender reações químicas!",
            "...para entender como funcionam aparelhos e circuitos!",
            "...com resolução de problemas complexos!"
        ],
        "Humanas": [
            "...com leitura, interpretação e produção de textos!",
            "...que desenvolvem empatia e cuidado com o outro!",
            "...que discutem comportamento humano e sociedade!",
            "...voltadas à comunicação e influência social!",
            "...que envolvem convencer pessoas com ideias bem construídas!",
            "...de voluntariado e projetos sociais!",
            "...focadas em bem-estar físico e saúde!",
            "...com artes, desenhos, textos ou música!",
            "...que envolvem motivar pessoas a adotarem hábitos mais saudáveis!",
            "...que envolvem comunicar ideias que inspirem cuidado, bem-estar e qualidade de vida!",
            "...que envolvem debates e expressão de opiniões!",
            "...que envolvem analisar discursos, narrativas e seus impactos na sociedade!"
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
            "Medicina/Psicologia/Odontologia": [0, 1, 2, 8, 9],
            "Educação Física": [1, 5,6, 8, 9],
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

    if st.button("Descobrir meu curso ideal"):
        if len(selecoes) != 5:
            st.warning("Selecione exatamente 5 características!")
        else:
            
            dados_treino = []
            rotulos = []
            
            for curso, indices in cursos_map.items():
                for _ in range(15):  
                    vetor = np.zeros(len(caracteristicas))
                    
                    for idx in indices:
                        vetor[idx] = 1.5
                    
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
            
            **Você tem perfil de {st.session_state.perfil}** e se encaixa melhor em:
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
            user_y = 0.3  
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

    if st.button("Voltar"):
        st.session_state.etapa = 1
        st.rerun()
