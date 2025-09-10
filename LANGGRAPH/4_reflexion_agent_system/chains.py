from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
# from langchain_openai import ChatOpenAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

parser = JsonOutputToolsParser(return_id=True)

# Actor Agent Prompt 
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

endpoint = "https://oai-ai-dt-dev-martec-001.openai.azure.com/"
apiVersion = "2025-01-01-preview"
deployment = "oai-ai-dt-dev-martec-001"
scope = "https://cognitiveservices.azure.com/.default"

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

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion') 

validator = PydanticToolsParser(tools=[AnswerQuestion])

# Revisor section

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# response = first_responder_chain.invoke({
#     "messages": [HumanMessage("AI Agents taking over content creation")]
# })s

# print(response)

