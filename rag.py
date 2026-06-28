from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
pdfs = DirectoryLoader("./pdfs", glob="*.pdf", loader_cls=PyPDFLoader).load()
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer, chunk_size=500, chunk_overlap=50)
pedacos = splitter.split_documents(pdfs)
embeddings = OllamaEmbeddings(model="bge-m3:567m")
vectorstore = FAISS.from_documents(embedding=embeddings, documents=pedacos)

modelo = ChatOpenAI(base_url= "http://127.0.0.1:1234/v1",model="qwen/qwen3-4b-2507",api_key= "xxx")
#modelo = OllamaLLM(model="qwen3.5:4b")
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido. \n\nContexto: \n{contexto}"),
        ("human", "{query}")
    ]
)
def rag(pergunta):
  cadeia = prompt | modelo | StrOutputParser()
  # retrieve
  trechos = retriever.invoke(pergunta)
  # argument
  contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
  # generate
  return cadeia.invoke({ "query": pergunta, "contexto": contexto})

pergunta = "Como fazer um seguro viagem?"
# Sem utilizar o Rag
resposta = modelo.invoke(pergunta)
print(resposta)
#Utilizando o Rag, para que a IA busque a resposta nos documentos
print(rag(pergunta))