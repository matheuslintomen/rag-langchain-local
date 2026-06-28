from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
numero_dias = 7
numero_criancas = 2
atividade = "praia"

modelo_prompt = PromptTemplate(template="""
Crie um roteiro de viagem de {dias} dias,
para uma familia com {numero_criancas} criancas,
que gostam de {atividade}
""")
prompt = modelo_prompt.format(dias=numero_dias,
    numero_criancas = numero_criancas,
    atividade=atividade)

modelo = ChatOpenAI( base_url= "http://127.0.0.1:1234/v1",model="qwen/qwen3-4b-2507",api_key= "xxx")
resposta = modelo.invoke(prompt)
print(resposta.content)