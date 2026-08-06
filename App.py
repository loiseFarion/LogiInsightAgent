import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from Tools import createTools
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Start the app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Assistente LogiInsight - Assistente de IA do centro de distribuição", layout="centered")
st.title("📦 Assistente LogiInsight")

# ---------------------------------------------------------------------------
# Tool description
# ---------------------------------------------------------------------------
st.info("""
Este assistente utiliza um agente de IA, criado com LangChain para responder perguntas sobre a
operação do Centro de Distribuição CD-01 da LogiInsight, combinando duas fontes de dados internas:

- 📊 **Base de estoque (CSV)**: mais de 5.000 SKUs, com quantidade disponível, quantidade reservada,
  localização (corredor/nível), fornecedor, status, lote, dimensões e mais.
- 📄 **Manual de Procedimentos Operacionais (PDF)**: fluxos de recebimento, armazenagem, picking,
  packing e expedição, além de regras de decisão para situações de exceção e protocolos de segurança.

Você pode:

- 📄 **Gerar relatórios automáticos**:
    - **Relatório de informações gerais**: apresenta a dimensão do DataFrame, nomes e tipos das colunas, contagem de dados nulos e duplicados, além de sugestões de tratamentos e análises adicionais.
    - **Relatório de estatísticas descritivas**: exibe valores como média, mediana, desvio padrão, mínimo e máximo; identifica possíveis outliers e sugere próximos passos com base nos padrões detectados.

- 🔎 **Fazer cálculos e consultas pontuais sobre os dados**: como "Qual é a média da coluna X?", "Quantos registros existem para cada categoria da coluna Y?", "Qual a correlação entre custo e preço de venda?".
             
- 📊 **Criar gráficos automaticamente** com base em perguntas em linguagem natural.

- 📦 **Analisar a saúde geral do estoque**: como "Quais produtos estão com estoque baixo?", "Quantos produtos estão em situação crítica?", "Qual categoria possui mais itens em estoque?", "Qual é o valor total do estoque?", "Existe excesso de estoque?"

- 🔍 **Localizar um produto específico**: como "Quais são os dados do SKU-1020?", "Onde está o produto com EAN 7891234567890?", "Qual é o fornecedor do Protetor Solar?"

- 📄 **Fazer perguntas sobre os procedimentos**: "O que fazer se o código de barras estiver ilegível?", "Como funciona
  a reserva de pedidos?", "Quem pode autorizar liberar um veículo sem conferência completa?"

Ideal para analistas, cientistas de dados e equipes que buscam agilidade e insights rápidos com apoio de IA.
""")

# ---------------------------------------------------------------------------
# Data upload: by default uses the project's fixed files, but the user can optionally upload their own files
# ---------------------------------------------------------------------------
standardCsvPath = os.path.join(os.path.dirname(__file__), "InventoryStock.csv")
standardPdfPath = os.path.join(os.path.dirname(__file__), "WarehouseProceduresManual.pdf")
folderUploads = os.path.join(os.path.dirname(__file__), "uploadedData", "uploads")

st.markdown("### 📁 Fonte de dados")
st.caption(
    "Por padrão, o assistente usa a base de estoque e o manual de procedimentos do CD-01 da LogiInsight. "
    "Se quiser, você pode enviar seus próprios arquivos para usar no lugar."
)

with st.expander("➕ Usar meus próprios arquivos (opcional)"):
    csvSent = st.file_uploader("Base de estoque (CSV)", type="csv", key="uploadCsv")
    pdfSent = st.file_uploader("Manual de procedimentos (PDF)", type="pdf", key="uploadPdf")

@st.cache_data(show_spinner="Carregando base de estoque...")
def loadDefaultStock():
    return pd.read_csv(standardCsvPath)

# ---------------------------------------------------------------------------
# CSV: uses the one sent by the person, if available; otherwise, uses the project's standard
# ---------------------------------------------------------------------------
if csvSent is not None:
    df = pd.read_csv(csvSent)
    st.success(f"Usando o arquivo enviado: {csvSent.name}")
else:
    df = loadDefaultStock()

# ---------------------------------------------------------------------------
# PDF: uses the one sent by the person, if available; otherwise, uses the project's default
# ---------------------------------------------------------------------------
if pdfSent is not None:
    os.makedirs(folderUploads, exist_ok=True)
    activePdfPath = os.path.join(folderUploads, pdfSent.name)
    with open(activePdfPath, "wb") as f:
        f.write(pdfSent.getbuffer())
    st.success(f"Usando o manual enviado: {pdfSent.name}")
else:
    activePdfPath = standardPdfPath

st.markdown("### 🔍 Amostra da base de estoque em uso")
st.dataframe(df.head())

# ---------------------------------------------------------------------------
# LLM could be Google API key, Groq, or another of your preference
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0)

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# llm = ChatGoogleGenerativeAI(
# model="gemini-3.6-flash",
# google_api_key=GEMINI_API_KEY,
#     temperature=0,
# )

# ---------------------------------------------------------------------------
# Tools (CSV + RAG from the PDF manual), already pointing to the files in use
# ---------------------------------------------------------------------------
tools = createTools(df, activePdfPath)

# ---------------------------------------------------------------------------
# Prompt react
# ---------------------------------------------------------------------------
dfHead = df.head().to_markdown()

promptReactPt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    partial_variables={"dfHead": dfHead},
    template="""
    Você é o assistente virtual do centro de distribuição LogiInsight e sempre responde em português.

    Você tem acesso a duas fontes de informação:

    1. Um dataframe pandas chamado `df` com a base de estoque. Aqui estão as primeiras linhas,
       obtidas com `df.head().to_markdown()`:

        {dfHead}

    2. O Manual de Procedimentos Operacionais do centro de distribuição, consultável através de
       uma ferramenta própria (Manual de Procedimentos).

    Responda às seguintes perguntas da melhor forma possível.

    Para isso, você tem acesso às seguintes ferramentas:

    {tools}

    Use o seguinte formato:

    Question: a pergunta de entrada que você deve responder
    Thought: você deve sempre pensar no que fazer
    Action: a ação a ser tomada, deve ser uma das [{tool_names}]
    Action Input: a entrada para a ação
    Observation: o resultado da ação
    ... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
    Thought: Agora eu sei a resposta final
    Final Answer: a resposta final para a pergunta de entrada original.
    Quando usar a ferramenta pythonCodeTool: formate sua resposta final de forma clara, em lista, com valores separados por vírgulas e duas casas decimais sempre que apresentar números.

    Comece!

    Question: {input}
    Thought: {agent_scratchpad}"""
)

# ---------------------------------------------------------------------------
# Agent and orchestrator: the agent is created with the LLM, tools, and prompt, the orchestrator executes the agent and handles errors
# ---------------------------------------------------------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=promptReactPt)
orchestrator = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,return_intermediate_steps=True)

# ---------------------------------------------------------------------------
# Fast actions
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## ⚡ Ações rápidas")

# General Information Report
if st.button("📄 Relatório de informações gerais", key="generalReportButton"):
    with st.spinner("Gerando relatório 📦"):
        response = orchestrator.invoke({"input": "Quero um relatório com informações sobre os dados"})
        st.session_state['generalReport'] = response["output"]

if 'generalReport' in st.session_state:
    with st.expander("Resultado: Relatório de informações gerais"):
        st.markdown(st.session_state['generalReport'])
        st.download_button(
            label="📥 Baixar relatório",
            data=st.session_state['generalReport'],
            file_name="generalInformationReport.md",
            mime="text/markdown"
        )

# Descriptive statistics report
if st.button("📄 Relatório de estatísticas descritivas", key="descriptiveStatisticsButton"):
    with st.spinner("Gerando relatório 📦"):
        response = orchestrator.invoke({"input": "Quero um relatório de estatísticas descritivas"})
        st.session_state['descriptiveStatistics'] = response["output"]

if 'descriptiveStatistics' in st.session_state:
    with st.expander("Resultado: Relatório de estatísticas descritivas"):
        st.markdown(st.session_state['descriptiveStatistics'])
        st.download_button(
            label="📥 Baixar relatório",
            data=st.session_state['descriptiveStatistics'],
            file_name="descriptiveStatisticsReport.md",
            mime="text/markdown"
        )

# Questions (data, procedures, or calculations; the agent decides which tool to use)
st.markdown("---")
st.markdown("## 💬 Pergunte ao assistente")
st.caption("Pergunte sobre os dados de estoque, cálculos ou sobre os procedimentos operacionais do manual.")
userQuestion = st.text_input("Digite sua pergunta (ex: 'Qual a média de produtos no estoque?', 'Quantos SKUs estão com status Baixo Estoque?' ou 'O que fazer se o código de barras estiver ilegível?')")
if st.button("Responder pergunta", key="answerQuestion"):
    with st.spinner("Consultando 📦"):
        response = orchestrator.invoke({"input": userQuestion})
        st.markdown(response["output"])

# Generating Charts
st.markdown("---")
st.markdown("## 📊 Criar gráfico com base em uma pergunta")
questionGraph = st.text_input("Digite o que deseja visualizar (ex: 'Crie um gráfico da quantidade disponível por categoria')")
if st.button("Gerar gráfico", key="generateGraph"):
    with st.spinner("Gerando o gráfico 📦"):
        response = orchestrator.invoke({"input": questionGraph})
        agentUsed = [step[0].tool for step in response.get("intermediate_steps", [])]
        if "Gerar Gráfico" not in agentUsed:
            st.warning("⚠️ Essa pergunta não é sobre geração de gráficos. Por favor, "
                       "faça esse tipo de pergunta no campo **Pergunte ao assistente**, "
                       "e use este campo apenas para pedidos de gráficos.")