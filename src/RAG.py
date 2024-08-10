import os
import vertexai
import requests
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from google.cloud import secretmanager
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
from langchain.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
import numpy as np
import pandas as pd
from neo4j_runway import Discovery, GraphDataModeler, IngestionGenerator, LLM, PyIngest
from dotenv import load_dotenv
from neo4j_runway import (Discovery, GraphDataModeler, 
                          IngestionGenerator, LLM, PyIngest)
from neo4j_runway.utils import test_database_connection
from dotenv import load_dotenv

load_dotenv()

def read_config(config_file="/app/config.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def access_secret_version(secret_version_id):
  client = secretmanager.SecretManagerServiceClient()
  response = client.access_secret_version(name=secret_version_id)
  return response.payload.data.decode('UTF-8')


class RetrievalDoc():

    """
    This class provides functionality for retrieving relevant information from medical documents using a vector database and a large language model.

    Attributes:
        retriever: The retriever object used to retrieve relevant documents from the vector database.
        path: The path to the medical document to be indexed.
        uri: The URI for accessing the database.
        password: The password for accessing the database.
        openai_key: The OpenAI API key.
        chunk_size: The size of each chunk of text to be indexed.
        overlap: The overlap between adjacent chunks of text.
        top_k: The number of top results to retrieve.
        PROJECT_ID: The Google Cloud Project ID.
        llm: The large language model (LLM) used for generating responses.
        __username: The username for the database.
        __database: The database name.

    Methods:
        create_index(): Creates an index of the medical document in the vector database.
        get_results(question): Retrieves relevant information from the vector database and generates a response using the LLM.
    """


    def __init__(self, chunk_size:int=2000,overlap:int=100,top_k:int=1,config=None):
        
        """
        Initializes the RetrievalDoc object.

        Args:
            chunk_size (int, optional): The size of each chunk of text to be indexed. Defaults to 2000.
            overlap (int, optional): The overlap between adjacent chunks of text. Defaults to 100.
            top_k (int, optional): The number of top results to retrieve. Defaults to 1.
            config (dict, optional): A dictionary containing configuration settings. Defaults to None.
        """

        self.retriever = None 
        self.path=None
        secret_version_uri = f"projects/789654123658/secrets/uri/versions/latest"
        uri = access_secret_version(secret_version_uri)
        self.uri = uri
        secret_version_password = f"projects/789654123658/secrets/password/versions/latest"
        password = access_secret_version(secret_version_password)
        self.password = password
        secret_version_openai_key = f"projects/789654123658/secrets/openai_key/versions/latest"
        openai_key = access_secret_version(secret_version_openai_key)
        self.openai_key = openai_key

        
        if config:
            self.path=config['PATH']
            self.llm = ChatVertexAI(model="gemini-1.5-flash")
            self.__username = config['username']
            self.__database = config['database']
            self.chunk_size = config['chunk_size']
            self.overlap = config['overlap']
            self.top_k = config['top_k']
            secret_version_id = f"projects/789654123658/secrets/PROJECT_ID/versions/latest"
            self.PROJECT_ID=access_secret_version(secret_version_id)
            vertexai.init(project=self.PROJECT_ID, location=config['LOCATION'])
            
            
    def create_index(self): 

        """
        Creates an index of the medical document in the vector database.
        """

        loader = UnstructuredPDFLoader(self.path) 
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(self.chunk_size), chunk_overlap=int(self.overlap))
        splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=splits,
        embedding=VertexAIEmbeddings(
        requests_per_minute=100,
        num_instances_per_batch=5,
        model_name = "textembedding-gecko"))
        self.retriever = vectorstore.as_retriever()
        
    
    def get_results(self, question):
        """
        Retrieves relevant information from the vector database and generates a response using the LLM.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The response generated by the LLM.
        """
        if not self.retriever:
            raise ValueError("Retriever has not been initialized. Please call create_index() first.")

        history=[]

        prompt_template = """"
        You are a medical expert and you have to analyze the medical report in the following context:
        {context}. 
        You can use the chat history: {chat_history} and {context}
        to answer users' question: {question}.
        If you don't know, please answer considering your knowledge base. Please be polite. """

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True, output_key='answer')
        messages = [
            SystemMessagePromptTemplate.from_template(prompt_template),
            HumanMessagePromptTemplate.from_template("{question}")
            ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)


        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm, self.retriever, memory=memory,combine_docs_chain_kwargs={"prompt": qa_prompt})
        answer = qa_chain({"question":question,"chat_history":history})
        history.append((question, answer))
        return  answer['answer']


class BuildKnowledgeGraph(RetrievalDoc):
    """
    This class extends RetrievalDoc to build a knowledge graph from a medical dataset.

    Attributes:
        disease_df: A Pandas DataFrame containing the medical data.
        data_path: The path to the CSV file containing the medical data.
        username: The username for the database.
        database: The database name.
        csv_dir: The directory for the CSV files.
        csv_name: The name of the CSV file.

    Methods:
        prepare_data(): Loads and prepares the medical data from the CSV file.
        generate_knowledge_graph_llm(config): Generates a knowledge graph from the medical data using a large language model (LLM).
    """
    def __init__(self, config):
        """
        Initializes the BuildKnowledgeGraph object.

        Args:
            config (dict): A dictionary containing configuration settings.
        """
        super().__init__(config=config) 
        self.disease_df = None

        if config:
            self.data_path=config['data_path']
            self.username = config['username']
            self.database = config['database']
            self.csv_dir=config['csv_dir'], 
            self.csv_name=config['csv_name']

    def prepare_data(self):
        """
        Loads and prepares the medical data from the CSV file.
        """
        disease_df = pd.read_csv(self.data_path)
        disease_df.columns = disease_df.columns.str.strip()
        for i in disease_df.columns:
            disease_df[i] = disease_df[i].astype(str)
        self.disease_df=disease_df

    def generate_knowledge_graph_llm(self,config):
        """
        Generates a knowledge graph from the medical data using a large language model (LLM).

        Args:
            config (dict): A dictionary containing configuration settings.
        """
        USER_GENERATED_INPUT = {
        'Disease': 'The name of the disease or medical condition.',
        'Fever': 'Indicates whether the patient has a fever (Yes/No).',
        'Cough': 'Indicates whether the patient has a cough (Yes/No).',
        'Fatigue': 'Indicates whether the patient experiences fatigue (Yes/No).',
        'Difficulty Breathing': 'Indicates whether the patient has difficulty breathing (Yes/No).',
        'Age': 'The age of the patient in years.',
        'Gender': 'The gender of the patient (Male/Female).',
        'Blood Pressure': 'The blood pressure level of the patient (Normal/High).',
        'Cholesterol Level': 'The cholesterol level of the patient (Normal/High).',
        'Outcome Variable': 'The outcome variable indicating the result of the diagnosis or assessment for the specific disease (Positive/Negative).'
        }
        llm = LLM(model="gpt-4o-2024-05-13", open_ai_key=self.openai_key)
        disc = Discovery(llm=llm, user_input=USER_GENERATED_INPUT, data=self.disease_df)
        disc.run()
        gdm = GraphDataModeler(llm=llm, discovery=disc)
        gdm.create_initial_model()
        gdm.iterate_model(user_corrections='''
        Let's think step by step. Please make the following updates to the data model:
        1. Remove the relationships between Patient and Disease, between Patient and Symptom and between Patient and Outcome.
        2. Change the Patient node into Demographics.
        2. Create a relationship HAS_DEMOGRAPHICS from Symptom to Demographics.
        3. Create a relationship HAS_SYMPTOM from Disease to Symptom. If the Symptom value is No, remove this relationship.
        4. Create a relationship HAS_LAB from Disease to HealthIndicator.
        5. Create a relationship HAS_OUTCOME from Disease to Outcome.
        ''')
        gen = IngestionGenerator(data_model=gdm.current_model, 
                         username=config['username'], 
                         password=self.password,
                         uri=self.uri,
                         database=config['database'], 
                         csv_dir=config['csv_dir'], 
                         csv_name=config['csv_name'])
        pyingest_yaml = gen.generate_pyingest_yaml_string()
        gen.generate_pyingest_yaml_file(file_name="disease_prepared")
        PyIngest(yaml_string=pyingest_yaml, dataframe=self.disease_df)


class KnowledgeGraphQuery(RetrievalDoc):
    """
    This class extends RetrievalDoc to query a knowledge graph using a large language model (LLM).

    Attributes:
        llm: The large language model (LLM) used to generate Cypher queries.

    Methods:
        get_query(config): Generates a Cypher query for a given question using the LLM and the knowledge graph schema.
    """
    def __init__(self, config):
        """
        Initializes the KnowledgeGraphQuery object.

        Args:
            config (dict): A dictionary containing configuration settings.
        """
        super().__init__(config=config)
        self.llm = ChatVertexAI(model="gemini-1.5-flash",temperature=config['temperature'],
        max_tokens=config['max_tokens'],
        max_retries=3)

        
    def get_query(self,config):
        """
        Generates a Cypher query for a given question using the LLM and the knowledge graph schema.

        Args:
            config (dict): A dictionary containing configuration settings.

        Returns:
            str: The generated Cypher query.
        """
        kg = Neo4jGraph(
        url=self.uri, username=config['username'], password=self.password, database=config['database'])
        kg.refresh_schema()
        schema=kg.schema
        prompt_template = """
        Let's think step by step:

        Step1: Task:
        Generate an effective and concise Cypher statement with less than 256 characteres to query a graph database
        Do not comment the code.

        Step 2: Get to know the database schema: {schema}

        Step 3: Instructions:
        - In the cypher query, ONLY USE the provided relationship types and properties that appear in the schema AND in the user question.
        - In the cypher query, do not use any other relationship types or properties in the user's question that are not contained in the provided schema.
        - Regarding Age, NEVER work with the age itself. For example: 24 years old, use interval: more than 20 years old.
        - USE ONLY ONE statement for Age, always use 'greater than', never 'less than' or 'equal'.
        - DO NOT USE property keys that are not in the database.

        Step 4: Examples: 
        Here are a few examples of generated Cypher statements for particular questions:

        4.1 Which diseases present high blood pressure?
        MATCH (d:Disease)
        MATCH (d)-[r:HAS_LAB]->(l)
        WHERE l.bloodPressure = 'High'
        RETURN d.name

        4.2 Which diseases present indicators as high blood pressure?
        // Match the Disease nodes
        MATCH (d:Disease)
        // Match HAS_LAB relationships from Disease nodes to Lab nodes
        MATCH (d)-[r:HAS_LAB]->(l)
        MATCH (d)-[r2:HAS_OUTCOME]->(o)
        // Ensure the Lab nodes have the bloodPressure property set to 'High'
        WHERE l.bloodPressure = 'High' AND o.result='Positive'
        RETURN d.name

        4.3 What is the name of a disease of the elderly where the patient presents high blood pressure, high cholesterol, fever, fatigue
        MATCH (d:Disease)
        MATCH (d)-[r1:HAS_LAB]->(lab)
        MATCH (d)-[r2:HAS_SYMPTOM]->(symptom)
        MATCH (symptom)-[r3:HAS_DEMOGRAPHICS]->(demo)
        WHERE lab.bloodPressure = 'High' AND lab.cholesterolLevel = 'High' AND symptom.fever = 'Yes' AND symptom.fatigue = 'Yes' AND TOINTEGER(demo.age) >40
        RETURN d.name

        4.4 What disease gives you fever, fatigue, no cough, no short breathe in people with high cholesterol?
        MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.fever = 'Yes' AND s.fatigue = 'Yes' AND s.difficultyBreathing = 'No' AND s.cough = 'No'
        MATCH (d:Disease)-[r1:HAS_LAB]->(lab:HealthIndicator)
        MATCH (d)-[r2:HAS_OUTCOME]->(o:Outcome)
        WHERE lab.cholesterolLevel='High' AND o.result='Positive'
        RETURN d.name


        Step 5. These are the values allowed for each entity:
        - Fever: Indicates whether the patient has a fever (Yes/No).
        - Cough: Indicates whether the patient has a cough (Yes/No).
        - Fatigue: Indicates whether the patient experiences fatigue (Yes/No).
        - Difficulty Breathing': 'Indicates whether the patient has difficulty breathing (Yes/No).
        - Age: The age of the patient in years.
        - Gender: The gender of the patient (Male/Female).
        - Blood Pressure: The blood pressure level of the patient (Normal/High).
        - Cholesterol Level: The cholesterol level of the patient (Normal/High).
        - Outcome Variable: The outcome variable indicating the result of the diagnosis or assessment for the specific disease (Positive/Negative).

        Step 6. Answer the question {question}."""
        cypher_prompt = PromptTemplate(
        input_variables=["schema","question"], 
        template=prompt_template)

        cypherChain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=kg,
            verbose=True,
            cypher_prompt=cypher_prompt,
            top_k=config['top_k'])
        
        cypherChain.run("What diseases may the patient have?")
