from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import datetime

load_dotenv()

endpoint = "https://oai-ai-dt-dev-martec-001.openai.azure.com/"
apiVersion = "2025-01-01-preview"
deployment = "oai-ai-dt-dev-martec-001"
scope = "https://cognitiveservices.azure.com/.default"
# if "AZURE_OPENAI_API_KEY" not in os.environ:
def create_azure_openai_model(endpoint: str, apiVersion: str, deployment: str, scope: str ) :
    """
    Creates an instance of AzureChatOpenAI with the specified configuration.
    """
    # Ensure that the environment variable is set or use DefaultAzureCredential
    model = AzureChatOpenAI(
        azure_endpoint=endpoint,  # Replace with your Azure OpenAI endpoint
        api_version=apiVersion,  # Use the latest API version
        deployment_name=deployment,  # Replace with your deployment name (e.g., "gpt-4", "gpt-35-turbo")
        azure_ad_token_provider=lambda: DefaultAzureCredential().get_token(scope).token,
        temperature=1,
        max_tokens=4000,
    )
    return model

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time
 
azure_openai = create_azure_openai_model(endpoint=endpoint, apiVersion=apiVersion, deployment=deployment, scope=scope)
print(azure_openai.invoke("Hello, how are you?").content)

llm = create_azure_openai_model(endpoint=endpoint, apiVersion=apiVersion, deployment=deployment, scope=scope)

search_tool = TavilySearchResults(search_depth = "basic")

tools = [search_tool, get_system_time]

agent = initialize_agent(tools = tools, llm = llm, agent = "zero-shot-react-description", verbose = True)

agent.invoke("Tell current date and calulatew how many days for diwali are there ")
 