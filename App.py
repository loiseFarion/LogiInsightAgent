import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from Tools import createTools

# Start the app
st.set_page_config(page_title="Assistente LogiInsight - Assistente de IA do centro de distribuição", layout="centered")
st.title()

# Tool description
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

- 📦 **Analisar a saúde geral do estoque**: como"Quais produtos estão com estoque baixo?", "Quantos produtos estão em situação crítica?", "Qual categoria possui mais itens em estoque?", "Qual é o valor total do estoque?", "Existe excesso de estoque?"

- 🔍 **Localizar um produto específico**: como "Quais são os dados do SKU-1020?", "Onde está o produto com EAN 7891234567890?", "Qual é o fornecedor do Protetor Solar?"

- 📄 **Fazer perguntas sobre os procedimentos**: "O que fazer se o código de barras estiver ilegível?", "Como funciona
  a reserva de pedidos?", "Quem pode autorizar liberar um veículo sem conferência completa?"

Ideal para analistas, cientistas de dados e equipes que buscam agilidade e insights rápidos com apoio de IA.
""")

# ---------------------------------------------------------------------------
# Data upload: by default uses the project's fixed files, but the user can optionally upload their own files.
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


# CSV: uses the one sent by the person, if available; otherwise, uses the project's standard.
if csvSent is not None:
    df = pd.read_csv(csvSent)
    st.success(f"Usando o arquivo enviado: {csvSent.name}")
else:
    df = loadDefaultStock()


# PDF: uses the one sent by the person, if available; otherwise, uses the project's default.
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