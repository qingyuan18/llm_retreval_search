from typing import Optional
import boto3
import json
from botocore.config import Config
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents.agent_types import AgentType
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
    CallbackManagerForChainRun
)
from langchain.llms.bedrock import Bedrock
import os
from langchain.agents.tools import Tool
from typing import Optional, List, Any
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.schema import BaseRetriever, Document
from langchain.utilities import SerpAPIWrapper

class BedrockModelWrapper(Bedrock):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = "\nHuman: \n" + prompt + "\nAssistant:"   ## Satisfy Bedrock-Claude prompt requirements
        return super()._call(prompt, stop, run_manager, **kwargs)

class FullContentRetriever(BaseRetriever):
    
    doc_type="txt"
    doc_path="./"
    def _get_content_type(doc_path:str):
        fullname=""
        for root, dirs, files in os.walk(path):
            for f in files:
                fullname = os.path.join(root, f)
                if "txt" in fullname:
                      self.doc_type="txt"
                if "pdf" in fullname:
                      self.doc_type="pdf"
                if "doc" in fullname:
                    self.doc_type="doc"
        self.doc_path = fullname
        
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        if self.doc_type == "doc":
            word_loader = Docx2txtLoader(self.doc_path)
            word_document = word_loader.load()
            return list(word_document)
        if self.doc_type == "txt":
            txt_loader = TextLoader(self.doc_path)
            txt_document = txt_loader.load()
            return list(txt_document)        
        elif self.doc_type == "pdf":
            pdf_loader = PyPDFLoader(self.doc_path)
            pdf_document = pdf_loader.load()
            return list(pdf_document)
        else:
            return []



def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
  
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]
        

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    client_kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID","")
    client_kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY","")
    
    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client



## for aksk bedrock
def get_bedrock_aksk(secret_name='chatbot_bedrock', region_name = "us-west-2"):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['BEDROCK_ACCESS_KEY'],secret['BEDROCK_SECRET_KEY']


ACCESS_KEY, SECRET_KEY=get_bedrock_aksk()
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"  # E.g. "us-east-1"
os.environ["AWS_PROFILE"] = "default"
os.environ["AWS_ACCESS_KEY_ID"]=ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]=SECRET_KEY


#新boto3 sdk只能session方式初始化bedrock
boto3_bedrock = get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

parameters_bedrock = {
    "max_tokens_to_sample": 2048,
    #"temperature": 0.5,
    "temperature": 0,
    #"top_k": 250,
    #"top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

bedrock_llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs=parameters_bedrock)
bedrock_llm_additional = BedrockModelWrapper(model_id="anthropic.claude-v2", 
                                          client=boto3_bedrock, 
                                          model_kwargs=parameters_bedrock)

memory = ConversationBufferWindowMemory(k=2)
system_message = SystemMessage(
    content=(
        "Do your best to answer the questions. "
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
    )
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
)

retriever = FullContentRetriever()
retriever._get_content_type("./docs")
retriever_tool = create_retriever_tool(
    retriever,
    "search_enterprise_documents",
    "useful for when you need to searches and returns documents regarding the user's question",
)
search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="useful for when you need to answer questions by searching the website",
)
custom_tool_list = [retriever_tool,search_tool]

agent_executor = initialize_agent(custom_tool_list, bedrock_llm_additional, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                  verbose=True,max_iterations=3,
                                  handle_parsing_errors=True,
                                  memory=memory,
                                  return_intermediate_steps=True)
agent_executor.agent.llm_chain.prompt.template=prompt