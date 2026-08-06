import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
from difflib import get_close_matches
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Load the Groq API key
# ---------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0)

# ---------------------------------------------------------------------------
# Load the Gemini API key
# ---------------------------------------------------------------------------
"""
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)
"""

WhDocumentPath = os.path.join(os.path.dirname(__file__), "InventoryStock", "WarehouseProceduresManual.pdf")

# ---------------------------------------------------------------------------
# Tool 1: General information about the dataframe
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Tool 2: Statistical report
# ---------------------------------------------------------------------------
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
        
        REGRAS OBRIGATÓRIAS DE FORMATO:
        - Use apenas markdown simples: títulos com #, negrito com **texto**, listas com -.
        - Nuncca use notação matemática ou LaTeX (proibido: $...$, símbolos de potência como ^, notação científica como 7.89×10^12).
        - Escreva todos os números por extenso, no formato brasileiro comum. Exemplo: em vez de 
          "7.890002×10^12", escreva "7.890.002.000.000" ou, se for um código EAN, apresente o valor 
          exato sem casas decimais nem notação científica.
        - Valores monetários devem seguir sempre o formato "R$ 243,66" (com espaço após o "R$", vírgula para decimais, ponto para milhar).
        - Nunca junte palavras sem espaço (exemplo: nunca escreva "medianade", sempre "mediana de").
        - Não use símbolos especiais de negrito duplicado ou aninhado (exemplo: nunca "**R$243,66**(mediana 
          de**R$244,09)", separe corretamente cada elemento com espaços e pontuação normal).
        """,
        input_variables=["question", "summary"]
    )

    cadeia = template_response | llm | StrOutputParser()
    response = cadeia.invoke({"question": question, "summary": descriptiveStatistics})

    return response.replace("$", "\\$")

# ---------------------------------------------------------------------------
# Tool 3: Chart Generator  Verificar aqui
# ---------------------------------------------------------------------------
@tool
def chartGenerator(question: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuário.
    A instrução pode conter pedidos como: 'Crie um gráfico da quantidade disponível por categoria','Plote a distribuição de peso_kg'
    ou "Plote a relação entre custo_unitario e preco_venda". Palavras-chave comuns que indicam o uso desta ferramenta incluem:
    'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição', 'represente graficamente', entre outros.
    """
    df = df.copy()

    # When two lines have the same 'product' but different SKU/EAN/supplier
    # a 'product_label' is created combining name + identifier
    # unique only for duplicate items. Items without duplicates keep
    # the original name, so as not to pollute the axis with identifiers
    # consumers.
    idIdentifier = ['sku', 'codigo_ean', 'ean', 'codigo', 'codigo_produto', 'id']
    idColumns = next((c for c in idIdentifier if c in df.columns), None)

    if 'produto' in df.columns:
        if idColumns:
            dup_mask = df['produto'].duplicated(keep=False)
            df['produto_label'] = df['produto']
            df.loc[dup_mask, 'produto_label'] = (df.loc[dup_mask, 'produto'] + ' (' + df.loc[dup_mask, idColumns].astype(str) + ')')
        else:
            df['produto_label'] = df['produto']

    columnsInfo = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])

    template_response = PromptTemplate(
        template="""
        Você é um especialista em visualização de dados. Sua tarefa é gerar **apenas o código Python** para plotar um gráfico com base na 
        solicitação do usuário.

        ## Solicitação do usuário:
        "{question}"

        ## Metadados do DataFrame:
        {columnsInfo}

        ## Instruções obrigatórias:
            1. Use as bibliotecas `matplotlib.pyplot` (como `plt`) e `seaborn` (como `sns`).
            2. Defina o tema com `sns.set_theme()`
            3. USE exclusivamente nomes de colunas que aparecem, exatamente como escritos (mesma grafia, maiúsculas/minúsculas
               e acentuação), na lista em "## Metadados do DataFrame" abaixo. É proibido usar qualquer nome de
               coluna genérico, abreviado, traduzido ou "adivinhado" (exemplo: não use 'preco' se a coluna real é 'preco_venda';
               não use 'quantidade' se a coluna real é 'quantidade_disponivel'; não use 'data' se a coluna real é
               'data_entrada' ou 'data_ultima_movimentacao'). Antes de escrever cada `df['coluna']`, confira se
               'coluna' está na lista de metadados. Se a solicitação mencionar algo que não corresponde a nenhuma coluna real,
               escolha a coluna existente mais semanticamente próxima, não invente um nome novo.
               O DataFrame já existe na variável `df` e está disponível no ambiente de execução. Não utilize `pd.read_csv`,
               `pd.read_excel`, `open()` ou qualquer outra função de carregamento de arquivo. Não redefina, recrie ou
               sobrescreva a variável `df`. Utilize-a diretamente como já está.
               
               3b. Gere apenas o(s) gráfico(s) que a solicitação do usuário pede explicitamente, um único `plt.figure()` /
               `plt.show()` na grande maioria dos casos. Não gere automaticamente uma análise cobrindo todos os tipos de
               gráfico (distribuição, categórica, comparação, relação, série temporal, entre outros) "só para exemplificar". Gere
               mais de um gráfico (exemplo: `plt.subplot`) apenas se a solicitação pedir explicitamente múltiplas visualizações
               (exemplo: "crie um gráfico de X e outro de Y", "compare X e Y lado a lado").
               
               3c. Nunca inclua comentários no código que listem exemplos alternativos de perguntas, sugestões do tipo
               "# Exemplo: ...", ou qualquer texto que não seja diretamente sobre o gráfico sendo gerado para esta
               solicitação específica. O código deve responder única e exclusivamente a: "{question}", não gere nada
               relacionado a outros possíveis pedidos, mesmo que pareçam relevantes ao contexto dos dados.
            
            4. Escolha o tipo de gráfico adequado conforme a análise solicitada (isto é um guia de qual função usar, não uma
               lista de gráficos para gerar todos de uma vez):
            - **Distribuição de variáveis numéricas**: `histplot`, `kdeplot`, `boxplot` ou `violinplot`
            - **Contagem de registros por categoria** (quantas linhas/itens existem em cada categoria, sem somar nenhuma
              métrica, exemplo: "quantos produtos existem por categoria", "quantidade de produtos cadastrados por fornecedor"):
              `countplot`
            - **Soma/média de uma métrica numérica por categoria** (quando a pergunta menciona uma coluna numérica
              específica agregada por grupo, exemplo: "quantidade disponível por categoria", "estoque total por categoria",
              "preço médio por fornecedor"): Não use `countplot`. Agregue com `df.groupby('coluna_categoria')['coluna_numerica'].sum()`
              (ou `.mean()` conforme o pedido) e depois plote com `sns.barplot`. `countplot` Nunca deve ser usado quando a
              pergunta cita uma coluna numérica específica a ser somada/mediada, isso é sinal de agregação, não de contagem.
            - **Relação entre variáveis**: `scatterplot` ou `lineplot`
            - **Séries temporais**: `lineplot`, com o eixo X formatado como datas
            5. Crie a figura antes de qualquer comando de plotagem utilizando `plt.figure(figsize=(8, 4))`.
            6. Adicione título e rótulos (`labels`) apropriados aos eixos. Os rótulos (`plt.xlabel`/`plt.ylabel`) devem
               sempre ser texto legível e formatado para humanos, nunca apresente o nome cru da coluna ou o valor padrão gerado
               pelo seaborn (exemplo: use `plt.ylabel('Quantidade Disponível')` em vez de deixar `plt.ylabel('quantidade_disponivel')`
               ou o padrão `'count'`; use `plt.xlabel('Categoria')` em vez de `'categoria'`). Substitua underscores por
               espaços e letra maiúscula na primeira letra de cada palavra relevante.
            7. Posicione o título à esquerda com `loc='left'`, deixe o `pad=20` e use `fontsize=14`.
            8. Ajuste a rotação dos ticks do eixo X conforme o conteúdo:
                - Se for um gráfico de barras/contagem com rótulos categóricos longos (produto, categoria, nome), use `plt.xticks(rotation=45, ha='right')`.
                - Se os rótulos forem curtos (menos de 10 caracteres) ou for um eixo numérico/temporal, use `plt.xticks(rotation=0)`.
            9. Após criar o gráfico, remova as bordas com `sns.despine()`, execute `plt.tight_layout()` e finalize com `plt.show()`, nessa ordem.
            10. Nunca utilize nomes, categorias ou valores específicos vistos na amostra de dados diretamente no código (exemplo: não escreva `df[df['produto'] == 'X']`). 
                Utilize sempre operações genéricas sobre o `df` completo, como `.nlargest()`, `.sort_values()`, `.groupby()`, entre outros.
            11. Quando a solicitação envolver "top N" (maiores, menores, mais vendidos, entre outros) de itens individuais (uma linha = um item real,
                exemplo: um produto identificado por SKU), use `.nlargest(N, coluna)` ou `.sort_values(coluna, ascending=False).head(N)`
                sobre o `df` inteiro, sem usar `groupby`. Cada linha do df já representa um item único mesmo que o nome se repita.
            12. Quando o usuário pedir um ranking por categoria agregada (exemplo: "categoria", "fornecedor", onde múltiplas linhas devem
                ser somadas/combinadas em um único valor por grupo), use `groupby()` + agregação e somente depois aplique
                `.nlargest()`/`.sort_values().head()`. Nunca aplique `.nlargest()` antes do `.groupby()` nesse caso.
            13. Ao usar `sns.barplot`, sempre passe `errorbar=None` (ou `ci=None`) para não exibir barras de erro, a menos que o usuário peça explicitamente por intervalo de confiança.
            14. Ao plotar um gráfico de barras a partir de um "top N" de itens individuais, nunca utilize `.index` do DataFrame
                como eixo categórico (x). Sempre utilize a coluna de rótulo apropriada (de acordo com a regra 15) como valor de `x`.
            15. Importante: se a coluna 'produto_label' existir no df, ela já foi pré-processada para remover a ambiguidade de nomes de produtos
                repetidos (produtos com mesmo nome mas SKU/EAN diferentes recebem o identificador entre parênteses no rótulo).
                Sempre que o ranking/gráfico for por 'produto', use 'produto_label' como eixo x em vez de 'produto', e:
                a. Passe `order=<subset>['produto_label']` (na ordem já ordenada) para o `sns.barplot`, garantindo que a ordem
                   do ranking seja preservada e que cada linha vire uma barra distinta mesmo com nomes repetidos.
                b. Defina `plt.xlabel('produto')` para manter o rótulo do eixo limpo, já que o nome da coluna usada é 'produto_label'.
            16. Nunca crie uma nova figura (`plt.figure()`) depois de chamar funções de plotagem (`sns.barplot`, `sns.scatterplot`, `plt.plot`, etc.). Toda configuração da figura deve ocorrer antes do primeiro gráfico.        
            17. Regra de rótulos de valor aplica-se somente quando o gráfico principal for `sns.barplot` ou
                `sns.countplot`. Nesses dois casos, capture o retorno da função em uma variável `ax`
                e, logo após criar o gráfico (antes do `sns.despine()`), adicione:
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)
                Exemplo completo:
                    ax = sns.barplot(x='categoria', y='quantidade_disponivel', data=df, errorbar=None)
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)
                Se os valores forem decimais (exemplo: preços, médias), use `fmt='%.2f'` em vez de `fmt='%.0f'`.
                Importante: para qualquer outro tipo de gráfico (`histplot`, `kdeplot`, `boxplot`, `violinplot`,
                `scatterplot`, `lineplot`), nunca use `ax.bar_label` nem crie a variável `ax` para esse fim, essas
                funções não possuem barras/containers e o código quebraria com `NameError` ou `AttributeError`. Nesses
                casos, apenas plote normalmente, sem tentar exibir valores individuais sobre os pontos/linhas/caixas.

            Retorne APENAS o código Python, sem nenhum texto adicional ou explicação.

            Código Python:
            """, 
            input_variables=["question", "columnsInfo"]
    )

    cadeia = template_response | llm | StrOutputParser()

    def extractReferencedColumns(code: str):
        colsQuotes = re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]", code)
        return set(colsQuotes)

    def generateCode(extraQuestion: str = ""):
        rawCode = cadeia.invoke({
            "question": question + extraQuestion,
            "columnsInfo": columnsInfo
        })
        return rawCode.replace("```python", "").replace("```", "").strip()

    cleanCode = generateCode()
    columnsUsed = extractReferencedColumns(cleanCode)
    columnsNonexistent = columnsUsed - set(df.columns)

    if columnsNonexistent:
        warningError = (
            f"\n\nATENÇÃO: na tentativa anterior você usou a(s) coluna(s) inexistente(s) "
            f"{sorted(columnsNonexistent)}, que NÃO existem no df. "
            f"Use exclusivamente colunas da lista de metadados fornecida. Gere o código novamente, corrigido."
        )
        cleanCode = generateCode(warningError)
        columnsUsed = extractReferencedColumns(cleanCode)
        columnsNonexistent = columnsUsed - set(df.columns)

    if columnsNonexistent:
        message = (
            f"Não foi possível gerar o gráfico: o modelo referenciou coluna(s) que não existem no DataFrame "
            f"({sorted(columnsNonexistent)}). Colunas disponíveis: {list(df.columns)}. "
            f"Tente reformular a pergunta especificando o nome exato da coluna."
        )
        st.error(message)
        return message

    st.code(cleanCode, language="python")
    execGlobals = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'st': st}
    execLocals = {}
    try:
        exec(cleanCode, execGlobals, execLocals)
    except (NameError, AttributeError) as e:
        filteredLines = [
            line for line in cleanCode.splitlines()
            if 'bar_label' not in line and 'ax.containers' not in line
            and not line.strip().startswith('for container in ax.containers')
        ]
        codeWithoutBarLabel = "\n".join(filteredLines).replace('ax = sns.', 'sns.')
        try:
            plt.clf()
            exec(codeWithoutBarLabel, execGlobals, execLocals)
        except Exception as e2:
            message = f"Erro ao executar o gráfico gerado: {e2}"
            st.error(message)
            return message
    except Exception as e:
        message = f"Erro ao executar o gráfico gerado: {e}"
        st.error(message)
        return message

    fig = plt.gcf()
    st.pyplot(fig)

    return ""

# ---------------------------------------------------------------------------
# Tool 4: Consult the Procedures Manual (RAG on the PDF)
# ---------------------------------------------------------------------------
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
        exceção (exemplo: "o que fazer se..."), destaque a ação esperada e o responsável pela decisão,
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

# ---------------------------------------------------------------------------
# Tool 5: Search by product
# ---------------------------------------------------------------------------
@tool
def searchProduct(question: str, df: pd.DataFrame) -> str:
    """
    Busca um produto específico no estoque por SKU, EAN ou nome, e responde à pergunta do usuário
    sobre esse produto, seja um dado pontual (exemplo: preço, localização, quantidade) ou uma ficha completa.

    Identificadores aceitos na pergunta:
    - SKU (exemplo: SKU-1234)
    - Código EAN
    - Nome ou parte do nome do produto (exemplo: "Protetor Solar")

    Não use para rankings, comparações ou análises envolvendo múltiplos produtos.
    """

    # ==================================================================
    # STEP 1: Locate the product in the dataframe
    # ==================================================================
    result, identifier = locateProduct(question, df)

    if result.empty:
        return f"Nenhum item encontrado para '{identifier}'. Verifique o SKU, EAN ou nome do produto."

    if len(result) > 1:
        skus = ", ".join(result["sku"].astype(str).tolist())
        return f"Encontrei {len(result)} itens para '{identifier}': {skus}. Especifique o SKU para ver a ficha completa."

    itemDict = result.iloc[0].to_dict()

    # ==================================================================
    # STEP 2: Generate response via LLM (decides format and responds)
    # ==================================================================
    return generateAnswer(question, itemDict)

def locateProduct(question: str, df: pd.DataFrame):
    """Tenta localizar o produto por SKU, depois EAN, depois nome (exato e fuzzy)."""

    skuMatch = re.search(r"SKU-\d+", question, re.IGNORECASE)
    if skuMatch:
        identifier = skuMatch.group().upper()
        return df[df["sku"].astype(str).str.strip().str.upper() == identifier], identifier

    eanMatch = re.search(r"\b\d{8,14}\b", question)
    if eanMatch:
        identifier = eanMatch.group().strip()
        eanNormalized = (
            df["codigo_ean"].astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.extract(r"(\d+)", expand=False)
            .fillna("")
        )
        return df[eanNormalized == identifier], identifier

    identifier = question.strip()
    result = df[df["produto"].str.contains(question, case=False, na=False, regex=False)]
    if not result.empty:
        return result, identifier

    products = df["produto"].dropna().astype(str).tolist()
    matches = get_close_matches(question, products, n=5, cutoff=0.5)
    return df[df["produto"].isin(matches)], identifier

def generateAnswer(question: str, item_dict: dict) -> str:
    """Chama o LLM uma única vez: ele decide se a pergunta é pontual ou geral e já responde."""

    context = "\n".join(f"{campo}: {valor}" for campo, valor in item_dict.items())

    template = PromptTemplate(
        template="""Você é um analista de dados de estoque. Avalie a pergunta do usuário sobre o produto abaixo:

        - Se pedir um dado pontual (localização, preço, quantidade, fornecedor, status, peso, lote, data, entre outros),
          responda em 1 a 2 frases diretas, citando apenas o(s) dado(s) pedido(s).
        - Se pedir uma visão geral ou ficha completa, gere uma ficha com:
          1. Título: ## Código SKU - Nome Produto - Categoria
          2. Visão geral do produto
          3. Seção por informação disponível, com interpretação objetiva
          4. Destaque de quantidade, status, localização, fornecedor, custo e preço

        Pergunta: {question}
        Dados completos: {item}

        REGRAS: use apenas dados disponíveis, não invente valores, não mencione seu processo, vá direto à resposta.""",
        input_variables=["question", "item"],
    )
    chain = template | llm | StrOutputParser()
    return chain.invoke({"question": question, "item": context})

# ---------------------------------------------------------------------------
# Tool 6: Analyze Stock
# ---------------------------------------------------------------------------
@tool
def analyzeStock(question: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário fizer perguntas analíticas sobre a saúde ou situação geral 
    do estoque. Exemplos de uso incluem: "Quais produtos estão com estoque baixo?", "Quantos produtos estão 
    em situação crítica?", "Qual categoria possui mais itens em estoque?", "Qual é o valor total do estoque?",
    "Quais produtos representam maior valor financeiro?", "Existe excesso de estoque?".

    Não utilize esta ferramenta para localizar um produto específico por SKU, EAN ou nome e para estatísticas 
    descritivas genéricas de todas as colunas numéricas.
    """
    
    df["valor_total_item"] = df["quantidade_disponivel"] * df["custo_unitario"]

    # 1. Low stock: available quantity equal to or below the defined minimum
    lowStock = df[df["quantidade_disponivel"] <= df["estoque_minimo"]]
    lowStockQuantity = len(lowStock)
    lowStockList = lowStock[["produto", "sku", "categoria", "quantidade_disponivel", "estoque_minimo"]].to_string(index=False)

    # 2. Critical situation: out of stock available
    critics = df[df["quantidade_disponivel"] == 0]
    criticalQuantity = len(critics)
    criticalList = critics[["produto", "sku", "categoria"]].to_string(index=False)

    # 3. Possible overage: quantity available far above the minimum (adjustable threshold)
    excess = df[df["quantidade_disponivel"] > 3 * df["estoque_minimo"]]
    excessQuantity = len(excess)
    excessList = excess[["produto", "sku", "categoria", "quantidade_disponivel", "estoque_minimo"]].to_string(index=False)

    # 4. Ranking of categories by total quantity in stock
    rankingCategories = df.groupby("categoria")["quantidade_disponivel"].sum().sort_values(ascending=False).to_string()

    # 5. Total stock value
    totalStockValue = df["valor_total_item"].sum()

    # 6. Top 10 products by financial value in stock
    topValue = df.nlargest(10, "valor_total_item")[["produto", "sku", "categoria", "quantidade_disponivel", "custo_unitario", "valor_total_item"]].to_string(index=False)

    template_response = PromptTemplate(
        template="""
        Você é um especialista em analista de estoque. Sua função é responder sobre o estoque de um centro de 
        distribuição a partir de uma {question} feita pelo usuário.

        A seguir estão os dados calculados sobre a situação geral do estoque. Utilize apenas as
        informações relevantes para a pergunta feita.

        ================= PRODUTOS COM ESTOQUE BAIXO ({lowStockQuantity} itens) =================
        {lowStockList}

        ================= PRODUTOS EM SITUAÇÃO CRÍTICA - SEM ESTOQUE ({criticalQuantity} itens) =================
        {criticalList}

        ================= POSSÍVEL EXCESSO DE ESTOQUE ({excessQuantity} itens) =================
        {excessList}

        ================= RANKING DE CATEGORIAS (quantidade total disponível) =================
        {rankingCategories}

        ================= VALOR TOTAL DO ESTOQUE =================
        R$ {totalStockValue:.2f}

        ================= TOP 10 PRODUTOS POR VALOR FINANCEIRO EM ESTOQUE =================
        {topValue}

        ============================================================

        Com base nesses dados, elabore uma resposta com linguagem clara, objetiva e fluida, incluindo:
        1. Um título relacionado a pergunta: ## Produtos com estoque baixo.
        2. A resposta direta à pergunta, citando os produtos e números relevantes.
        3. Um breve comentário sobre o impacto operacional: risco de ruptura, capital parado, necessidade 
           de reposição.
        4. Se a pergunta for genérica por exemplo "como está o estoque?", apresente uma visão geral cobrindo
           estoque baixo, críticos, excesso e valor total.
        """,

        input_variables=["question", "lowStockQuantity", "lowStockList", "criticalQuantity", "criticalList", "excessQuantity", "excessList", "rankingCategories", "totalStockValue", "topValue"]
    )

    cadeia = template_response | llm | StrOutputParser()

    response = cadeia.invoke({
        "question": question,
        "lowStockQuantity": lowStockQuantity,
        "lowStockList": lowStockList if lowStockQuantity else "Nenhum produto com estoque baixo.",
        "criticalQuantity": criticalQuantity,
        "criticalList": criticalList if criticalQuantity else "Nenhum produto em situação crítica.",
        "excessQuantity": excessQuantity,
        "excessList": excessList if excessQuantity else "Nenhum produto identificado com excesso.",
        "rankingCategories": rankingCategories,
        "totalStockValue": totalStockValue,
        "topValue": topValue
    })

    return response

# ---------------------------------------------------------------------------
# Function to create the agent tools
# ---------------------------------------------------------------------------
def createTools(df, WhDocumentPath=WhDocumentPath):
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
                    resumos estatísticos completos ou geração de gráficos, nesses casos, use as ferramentas apropriadas.
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
                    SKUs específicos ou gráficos, nesses casos, use as ferramentas apropriadas.
                    """,
        return_direct=True)

    productFindTool = Tool(
        name="Localizar Produto",
        func=lambda question: searchProduct.run({"question": question, "df": df}),
        description="""
                    Utilize esta ferramenta sempre que o usuário quiser localizar um produto específico no estoque. 
                    Exemplos de uso: "Quais são os dados do SKU-1020?", "Onde está o produto com EAN 7891234567890?", 
                    "Qual é o fornecedor do Protetor Solar?" A ferramenta recebe a pergunta do usuário, localiza o 
                    item correspondente na base de estoque e retorna  suas informações. Se o produto não for encontrado, 
                    informe que ele não foi localizado na base de dados. 
                    Não utilize informações de outros produtos para responder à solicitação. Não utilize esta ferramenta para 
                    consultas que envolvam múltiplos produtos, análises gerais de estoque, estatísticas, rankings ou geração 
                    de gráficos.
                    """,
        return_direct=True)

    stockAnalysisTool = Tool(
        name="Analisar Estoque",
        func=lambda question: analyzeStock.run({"question": question, "df": df}),
        description="""
                    Utilize esta ferramenta sempre que o usuário fizer perguntas sobre a saúde ou situação geral do
                    estoque, como produtos com estoque baixo, situação crítica, ranking de categorias,
                    valor total do estoque, produtos de maior valor financeiro ou excesso de estoque.
                    Não utilize para localizar um produto específico ou para estatísticas descritivas
                    genéricas de colunas numéricas.
                    """,
        return_direct=True)
    
    return [
        dataframeInformationtool,
        statisticalSummaryTool,
        generateGraphTool,
        pythonCodeTool,
        procedureManualTool,
        productFindTool,
        stockAnalysisTool
    ]