from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
# from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential

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

llm = create_azure_openai_model(endpoint=endpoint, apiVersion=apiVersion, deployment=deployment, scope=scope)


class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: BasicChatBot):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])], 
    }

def tools_router(state: BasicChatBot):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    

tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatBot)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

app = graph.compile()

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print(result)





