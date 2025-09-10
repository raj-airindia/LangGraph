from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential


# if "AZURE_OPENAI_API_KEY" not in os.environ:
def create_azure_openai_model() :
    """
    Creates an instance of AzureChatOpenAI with the specified configuration.
    """
    # Ensure that the environment variable is set or use DefaultAzureCredential
    endpoint = "https://oai-ai-dt-dev-martec-001.openai.azure.com/"
    apiVersion = "2025-01-01-preview"
    deployment = "oai-ai-dt-dev-martec-001"
    scope = "https://cognitiveservices.azure.com/.default"
    model = AzureChatOpenAI(
        azure_endpoint=endpoint,  # Replace with your Azure OpenAI endpoint
        api_version=apiVersion,  # Use the latest API version
        deployment_name=deployment,  # Replace with your deployment name (e.g., "gpt-4", "gpt-35-turbo")
        azure_ad_token_provider=lambda: DefaultAzureCredential().get_token(scope).token,
        temperature=1,
        max_tokens=4000,
    )
    return model