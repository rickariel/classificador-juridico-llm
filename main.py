import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import json

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)

prompt = PromptTemplate(
    input_variables=["texto"],
    template="""
Classifique o seguinte texto jurídico em uma das categorias:
- Petição Inicial
- Sentença
- Despacho
- Acordo
- Outro

Retorne APENAS JSON nesta estrutura:

{{
  "tipo": "...",
  "justificativa": "..."
}}

Texto:
{texto}
"""
)

def classificar_documento(texto):
    mensagem = HumanMessage(content=prompt.format(texto=texto))
    resposta = llm.invoke([mensagem])
    return json.loads(resposta.content)

if __name__ == "__main__":
    exemplo = "Vistos, etc. Trata-se de ação de indenização..."
    print(classificar_documento(exemplo))

