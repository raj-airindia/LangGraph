# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub


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

llm =  create_azure_openai_model(endpoint=endpoint, apiVersion=apiVersion, deployment=deployment, scope=scope)

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

import json

@tool
def calculate_days_between_dates(json_input: str, date_format: str = "%Y-%m-%d"):
    """
    Calculates the number of days between two dates from a JSON input.
    
    Args:
        json_input (str): A JSON string containing 'date1' and 'date2' fields.
        date_format (str): The format of the input dates (default is "%Y-%m-%d").
    
    Returns:
        int: The number of days between the two dates.

    example json input format below
        {
        "date1": "2025-07-08",
        "date2": "2025-07-15"
        }
    """
    try:
        # Parse the JSON input
        data = json.loads(json_input)
        date1 = data.get("date1")
        date2 = data.get("date2")
        
        # Validate that both dates are provided
        if not date1 or not date2:
            return "Error: Both 'date1' and 'date2' fields are required in the JSON input."
        
        # Convert dates to datetime objects
        start = datetime.datetime.strptime(date1, date_format)
        end = datetime.datetime.strptime(date2, date_format)
        
        # Calculate the absolute difference in days
        delta = abs(end - start)
        return delta.days
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input. {e}"
    except ValueError as e:
        return f"Error: {e}. Please ensure the dates are in the correct format ({date_format})."

search_tool = TavilySearchResults(search_depth="basic")
react_prompt = hub.pull("hwchase17/react")


tools = [get_system_time, search_tool, calculate_days_between_dates]

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)