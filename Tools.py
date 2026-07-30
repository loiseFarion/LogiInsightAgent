import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the Groq API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM Settings
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0
)

WhDocumentPath = os.path.join(os.path.dirname(__file__), "InventoryStock", "WarehouseProceduresManual.pdf")

# Tool 1: General information about the dataframe
@tool
def dataframeInformaion(question: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
    incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
    nulos e duplicados para dar um panorama geral sobre o arquivo.
    """

    shape = df.shape
    columns = df.dtypes
    nulls = df.isnull().sum()
    nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
    duplicates = df.duplicated().sum()

    template_response = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de apresentar um resumo informativo sobre um DataFrame
        de estoque de um centro de distribuição, a partir de uma {question} feita pelo usuário.

        A seguir, você encontrará as informações gerais da base de dados:

        ================= INFORMAÇÕES DO DATAFRAME =================

        Dimensões: {shape}

        Colunas e tipos de dados: {columns}

        Valores nulos por coluna: {nulls}

        Strings 'nan' (qualquer capitalização) por coluna: {nans_str}

        Linhas duplicadas: {duplicates}

        ============================================================

        Com base nessas informações, escreva um resumo claro e organizado contendo:

        1. Um título: ## Relatório de informações gerais sobre o dataset
        2. A dimensão total do DataFrame;
        3. A descrição de cada coluna (incluindo nome, tipo de dado e o que aquela coluna representa
           no contexto de um estoque de centro de distribuição — ex: quantidade_disponivel,
           quantidade_reservada, estoque_minimo, corredor, nivel_prateleira, status, fragil, lote etc.)
        4. As colunas que contêm dados nulos, com a respectiva quantidade.
        5. As colunas que contêm strings 'nan', com a respectiva quantidade.
        6. E a existência (ou não) de dados duplicados.
        7. Escreva um parágrafo sobre análises que podem ser feitas com esses dados.
        8. Escreva um parágrafo sobre tratamentos que podem ser feitos nos dados.
        """,
        input_variables=["question", "shape", "columns", "nulls", "nans_str", "duplicates"]
    )

    cadeia = template_response | llm | StrOutputParser()

    response = cadeia.invoke({
        "question": question,
        "shape": shape,
        "columns": columns,
        "nulls": nulls,
        "nans_str": nans_str,
        "duplicates": duplicates
    })

    return response

# Tool 2: Statistical report
@tool
def statisticalReport(question: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo e descritivo da base de dados,
    incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.).
    Não utilize esta ferramenta para calcular uma única métrica como 'qual é a média de X' ou 'qual a correlação das variáveis'.
    """
    descriptiveStatistics = df.describe(include='number').transpose().to_string()

    template_response = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de interpretar resultados estatísticos da base de estoque
        de um centro de distribuição, a partir de uma {question} feita pelo usuário.

        A seguir, você encontrará as estatísticas descritivas da base de dados:

        ================= ESTATÍSTICAS DESCRITIVAS =================

        {summary}

        ============================================================

        Com base nesses dados, elabore um resumo explicativo com linguagem clara, acessível e fluida, destacando
        os principais pontos dos resultados. Inclua:

        1. Um título: ## Relatório de estatísticas descritivas
        2. Uma visão geral das estatísticas das colunas numéricas
        3. Um parágrafo sobre cada uma das colunas, comentando informações sobre seus valores.
        4. Identificação de possíveis outliers com base nos valores mínimo e máximo
        5. Recomendações de próximos passos na análise com base nos padrões identificados
        """,
        input_variables=["question", "summary"]
    )

    cadeia = template_response | llm | StrOutputParser()
    response = cadeia.invoke({"question": question, "summary": descriptiveStatistics})

    return response

# Tool 3: Chart Generator
@tool
def chartGenerator(question: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuário.
    A instrução pode conter pedidos como: 'Crie um gráfico da quantidade disponível por categoria','Plote a distribuição de peso_kg'
    ou "Plote a relação entre custo_unitario e preco_venda". Palavras-chave comuns que indicam o uso desta ferramenta incluem:
    'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição', 'represente graficamente', entre outros.
    """

    columnsInfo = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    sampleData = df.head(3).to_dict(orient='records')

    template_response = PromptTemplate(
        template="""
        Você é um especialista em visualização de dados. Sua tarefa é gerar **apenas o código Python** para plotar um gráfico com base na solicitação do usuário.

        ## Solicitação do usuário:
        "{question}"

        ## Metadados do DataFrame:
        {columnsInfo}

        ## Amostra dos dados (3 primeiras linhas):
        {sampleData}

        ## Instruções obrigatórias:
        1. Use as bibliotecas `matplotlib.pyplot` (como `plt`) e `seaborn` (como `sns`).
        2. Defina o tema com `sns.set_theme()`
        3. Certifique-se de que todas as colunas mencionadas na solicitação existem no DataFrame chamado `df`.
        4. Escolha o tipo de gráfico adequado conforme a análise solicitada:
        - **Distribuição de variáveis numéricas**: `histplot`, `kdeplot`, `boxplot` ou `violinplot`
        - **Distribuição de variáveis categóricas**: `countplot`
        - **Comparação entre categorias**: `barplot`
        - **Relação entre variáveis**: `scatterplot` ou `lineplot`
        - **Séries temporais**: `lineplot`, com o eixo X formatado como datas
        5. Configure o tamanho do gráfico com `figsize=(8, 4)`.
        6. Adicione título e rótulos (`labels`) apropriados aos eixos.
        7. Posicione o título à esquerda com `loc='left'`, deixe o `pad=20` e use `fontsize=14`.
        8. Mantenha os ticks eixo X sem rotação com `plt.xticks(rotation=0)`
        9. Remova as bordas superior e direita do gráfico com `sns.despine()`.
        10. Finalize o código com `plt.show()`.

        Retorne APENAS o código Python, sem nenhum texto adicional ou explicação.

        Código Python:
        """, input_variables=["question", "columnsInfo", "sampleData"]
    )

    cadeia = template_response | llm | StrOutputParser()
    rawCode = cadeia.invoke({
        "question": question,
        "columnsInfo": columnsInfo,
        "sampleData": sampleData
    })

    cleanCode = rawCode.replace("```python", "").replace("```", "").strip()

    execGlobals = {'df': df, 'plt': plt, 'sns': sns}
    execLocals = {}
    exec(cleanCode, execGlobals, execLocals)

    fig = plt.gcf()
    st.pyplot(fig)

    return ""

# Tool 4: Consult the Procedures Manual (RAG on the PDF)
@st.cache_resource(show_spinner="Indexando o manual de procedimentos...")
def loadManualProcedureBase(WhDocumentPath: str = WhDocumentPath):
    """
    Carrega o PDF do manual, divide em pedaços (chunks) e cria o índice vetorial (FAISS).
    Fica em cache por caminho de arquivo, então cada PDF diferente (o padrão do projeto ou um
    enviado pelo usuário) só é processado uma vez.
    """

    loader = PyPDFLoader(WhDocumentPath)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    # Local embeddings using Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_documents(chunks,embeddings)
    return vectorstore

def consultProcedureManual(question: str, WhDocumentPath: str = WhDocumentPath) -> str:
    """
    Busca no manual de procedimentos os trechos mais relevantes para a pergunta e usa o LLM
    para responder com base apenas nesses trechos (RAG).
    """

    vectorstore = loadManualProcedureBase(WhDocumentPath)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevantDocuments = retriever.invoke(question)
    context = "\n\n---\n\n".join([doc.page_content for doc in relevantDocuments])

    template_response = PromptTemplate(
        template="""
        Você é um assistente especializado no Manual de Procedimentos Operacionais do centro de
        distribuição LogiInsight. Responda à pergunta do usuário utilizando SOMENTE as informações
        do trecho do manual fornecido abaixo. Se a resposta não estiver no trecho fornecido, diga
        claramente que a informação não foi encontrada no manual.

        ================= TRECHO DO MANUAL =================
        {context}
        ======================================================

        Pergunta: {question}

        Responda de forma clara e direta, em português. Se a resposta envolver uma situação de
        exceção (ex: "o que fazer se..."), destaque a ação esperada e o responsável pela decisão,
        quando essa informação estiver disponível no trecho.
        """,
        input_variables=["context", "question"]
    )

    cadeia = template_response | llm | StrOutputParser()

    response = cadeia.invoke({
        "context": context,
        "question": question
    })

    return response

# Function to create the agent tools
def criar_ferramentas(df):
    dataframeInformationtool = Tool(
        name="Informações Dataframe",
        func=lambda question:dataframeInformaion.run({"question": question, "df": df}),
        description="""
                    Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
                    incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
                    nulos e duplicados para dar um panomara geral sobre o arquivo.
                    """,
        return_direct=True)

    statisticalSummaryTool = Tool(
        name="Resumo Estatístico",
        func=lambda question:statisticalReport.run({"question": question, "df": df}),
        description="""
                    Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo e descritivo 
                    da base de dados, incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.) e/ou 
                    múltiplas colunas numéricas. Não utilize esta ferramenta para calcular uma única métrica como 
                    'qual é a média de X' ou 'qual a correlação das variáveis'. Para isso, use a ferramenta_python.
                    """,
        return_direct=True)

    generateGraphTool = Tool(
        name="Gerar Gráfico",
        func=lambda question:chartGenerator.run({"question": question, "df": df}),
        description="""
                    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas 
                    (`df`) com base em uma instrução do usuário. A instrução pode conter pedidos como: 'Crie um gráfico
                    da média de tempo de entrega por clima','Plote a distribuição do tempo de entrega'" ou "Plote a 
                    relação entre a classificação dos agentes e o tempo de entrega. Palavras-chave comuns que indicam o 
                    uso desta ferramenta incluem: 'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre 
                    a distribuição', 'represente graficamente', entre outros.
                    """,
        return_direct=True)

    pythonCodeTool = Tool(
        name="Códigos Python",
        func=PythonAstREPLTool(locals={"df": df}),
        description="""
                    Utilize esta ferramenta sempre que o usuário solicitar cálculos, consultas ou transformações específicas
                    usando Python diretamente sobre o DataFrame `df`. Exemplos de uso incluem: "Qual é a média da coluna X?",
                    "Quais são os valores únicos da coluna Y?", "Qual a correlação entre A e B?". Evite utilizar esta 
                    ferramenta para solicitações mais amplas ou descritivas, como informações gerais sobre o dataframe, 
                    resumos estatísticos completos ou geração de gráficos — nesses casos, use as ferramentas apropriadas.
                    """)
    
    procedureManualTool = Tool(
        name="Manual de Procedimentos",
        func=lambda question: consultProcedureManual(question, WhDocumentPath),
        description="""
                    Utilize esta ferramenta sempre que o usuário perguntar sobre PROCESSOS, REGRAS, FLUXOS ou DECISÕES
                    operacionais do centro de distribuição. Exemplos de uso incluem: "O que fazer se o código de barras 
                    estiver ilegível?", "Qual o procedimento de recebimento de mercadorias?", "Quem pode autorizar a 
                    liberação de um veículo sem conferência completa?", "Como funciona a reserva de pedidos?", "Quais EPIs 
                    são obrigatórios no setor de picking?". Não utilize esta ferramenta para perguntas sobre quantidades, 
                    SKUs específicos ou gráficos — nesses casos, use as ferramentas apropriadas.
                    """,
        return_direct=True)
    
    return [
        dataframeInformationtool,
        statisticalSummaryTool,
        generateGraphTool,
        pythonCodeTool,
        procedureManualTool
    ]