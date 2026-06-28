# RAG com LangChain e IA Local

Experimentos com LangChain, RAG (Retrieval-Augmented Generation) e modelos de linguagem rodando localmente via LM Studio.

## Arquivos

### `pergunta-ialocal.py`
Exemplo básico de como enviar um prompt para uma IA local atribuindo roles à conversa. Usa a biblioteca `openai` diretamente, sem LangChain, definindo um papel de sistema (assistente de roteiro de viagens) e uma mensagem do usuário.

### `langchain-basico.py`
Mesmo conceito do arquivo anterior, mas utilizando LangChain com `PromptTemplate` para estruturar o prompt com variáveis dinâmicas. Demonstra como criar templates reutilizáveis e invocar um modelo local via `ChatOpenAI`.

### `rag.py`
Implementação completa de RAG. Carrega PDFs de uma pasta local, divide os textos em pedaços usando um tokenizer da HuggingFace, gera embeddings com Ollama e armazena em um vectorstore FAISS. Na hora de responder, recupera os trechos mais relevantes dos documentos e os usa como contexto para a IA, limitando a resposta ao conteúdo fornecido.

## Requisitos

- Python 3.10+
- LM Studio rodando localmente na porta 1234
- Ollama rodando localmente com o modelo `bge-m3:567m`

## Instalação

```bash
pip install langchain langchain-community langchain-openai langchain-ollama langchain-text-splitters transformers faiss-cpu openai pypdf
```

## Como usar

Adicione seus PDFs em uma pasta chamada `pdfs` na raiz do projeto e execute:

```bash
python rag.py
```

## Configuração

Os scripts conectam por padrão em `http://127.0.0.1:1234/v1`. Caso seu servidor LLM esteja em outra porta, altere a variável `base_url` no topo de cada arquivo.