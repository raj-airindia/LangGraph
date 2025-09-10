
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential

endpoint = "https://oai-ai-dt-dev-martec-001.openai.azure.com/"
token_url = "https://cognitiveservices.azure.com/.default"
deployment = "text-embedding-ada-002"
api_version = "2023-05-15"

def get_azure_ad_token():
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token
 
def get_azure_embedding():
    embedding_function = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint,
    api_version=api_version,
    model=deployment,
    azure_ad_token_provider=get_azure_ad_token,
    )
    return embedding_function
 
 
print(get_azure_embedding())
