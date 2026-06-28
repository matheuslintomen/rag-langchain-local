from openai import OpenAI

numero_dias = 2
numero_criancas = 2
atividade = "sightseeing"
prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma familia com {numero_criancas} crianças, que gosta de {atividade}"

cliente = OpenAI(base_url= "http://127.0.0.1:1234/v1",
                 api_key= "xxx")
resposta = cliente.chat.completions.create(
    model="qwen/qwen3-1.7b",
    messages=[
        {"role": "system","content":"Você é um assistente de roteiro de viagens para Paris."},
        {"role": "user","content": prompt}
        ],
         temperature= 1.0
)

print(resposta)