from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
# from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

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

load_dotenv()

memory = MemorySaver()

# llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict): 
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState): 
    return {
       "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)

graph.add_edge("chatbot", END)

graph.set_entry_point("chatbot")

app = graph.compile(checkpointer=memory)

config = {"configurable": {
    "thread_id": 2
}}

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        print("AI: " + result["messages"][-1].content)

