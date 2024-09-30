import os
import json
import ast
# from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
# from llama_index import SimpleDirectoryReader
from pandasai import SmartDataframe
from pandasai.llm import BedrockClaude
import pandas as pd
# from langchain.llms.openai import OpenAI
# from langchain.chat_models import ChatOpenAI
from transformers import pipeline
import time
import boto3
import threading
import requests
import duckdb
from dotenv import load_dotenv
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional

from llama_index.core import ( 
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.settings import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding, Models
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser import SentenceSplitter



from typing import Sequence, Any, List
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
import logging
from llama_index.core.schema import BaseNode, Document , ObjectType , TextNode
from llama_index.core.constants import DEFAULT_CHUNK_SIZE
from llama_index.core.node_parser.text.sentence import SENTENCE_CHUNK_OVERLAP

class SafeSemanticSplitter(SemanticSplitterNodeParser):

    safety_chunker : SentenceSplitter = SentenceSplitter(chunk_size=DEFAULT_CHUNK_SIZE*4,chunk_overlap=SENTENCE_CHUNK_OVERLAP)

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes : List[BaseNode] = super()._parse_nodes(nodes=nodes,show_progress=show_progress,**kwargs)
        all_good = True
        for node in all_nodes:
            if node.get_type()==ObjectType.TEXT:
                node:TextNode=node
                if self.safety_chunker._token_size(node.text)>self.safety_chunker.chunk_size:
                    logging.info("Chunk size too big after semantic chunking: switching to static chunking")
                    all_good = False
                    break
        if not all_good:
            all_nodes = self.safety_chunker._parse_nodes(nodes,show_progress=show_progress,**kwargs)
        return all_nodes 



import matplotlib.pyplot as plt
plt.ion()

st_time=time.time()
file_paths = ["/tmp/data/csv/DataSet.csv", "/tmp/data/plans/Plans.pdf", "/tmp/data/column/Column Descriptions.pdf"]

# Check if all files exist
all_files_exist = all(os.path.exists(file_path) for file_path in file_paths)

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials from environment variables
aws_access_key_id = os.getenv('aws_access_key_id')
aws_secret_access_key = os.getenv('aws_secret_access_key')
aws_session_token = os.getenv('aws_session_token')

# Create a Bedrock Runtime client
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token
)

def download_file(bucket_name, file_name, destination_folder):
    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        s3.download_file(bucket_name, file_name, f"{destination_folder}/{file_name}")
        print(f"File '{file_name}' downloaded successfully to '{destination_folder}'.")
    except Exception as e:
        print(f"Error downloading file '{file_name}' from bucket '{bucket_name}': {e}")

def download_files_from_s3(bucket_name, files):
    threads = []
    for file_info in files:
        file_name, destination_folder = file_info
        thread = threading.Thread(target=download_file, args=(bucket_name, file_name, destination_folder))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def download_chunk(bucket_name, object_key, range_start, range_end, output_file_path):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=object_key, Range=f"bytes={range_start}-{range_end}")
    chunk_data = response['Body'].read()

    # Handle file locally within the process
    with open(output_file_path, 'r+b') as file:
        file.seek(range_start)
        file.write(chunk_data)

def download_file_in_chunks(bucket_name, object_key, num_chunks):
    s3 = boto3.client('s3')
    response = s3.head_object(Bucket=bucket_name, Key=object_key)
    total_size = response['ContentLength']

    chunk_size = total_size // num_chunks
    ranges = [(i * chunk_size, min((i + 1) * chunk_size - 1, total_size - 1)) for i in range(num_chunks)]

    output_file_path = '/tmp/data/bart-large-mnli/pytorch_model.bin'

    # Pre-initialize the file to ensure it is large enough
    with open(output_file_path, 'wb') as file:
        file.truncate(total_size)

    # Create threads for each chunk download
    threads = []
    for range_start, range_end in ranges:
        thread = threading.Thread(target=download_chunk, args=(bucket_name, object_key, range_start, range_end, output_file_path))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return output_file_path

if not all_files_exist:
    bucket_name = "bot-catalog"
    # file_name = "Column Descriptions.pdf"
    # destination_folder = "/tmp/data/column"
    if not os.path.exists("/tmp/data/bart-large-mnli"):
            os.makedirs("/tmp/data/bart-large-mnli")
    object_key = 'pytorch_model.bin'
    num_chunks = 10  # You can adjust the number of chunks as needed
    # download_file_in_chunks(bucket_name, object_key, num_chunks)

    s3 = boto3.client('s3')
    files = [("DataSet.csv", "/tmp/data/csv"), ("Plans.pdf", "/tmp/data/plans"), ("Column Descriptions.pdf", "/tmp/data/column"),("config.json","/tmp/data/bart-large-mnli"),("merges.txt","/tmp/data/bart-large-mnli"),("tokenizer_config.json","/tmp/data/bart-large-mnli"),("tokenizer.json","/tmp/data/bart-large-mnli"),("vocab.json","/tmp/data/bart-large-mnli"),("config.json","/tmp/data/bart-large-mnli")]
    
    files = [("DataSet.csv", "/tmp/data/csv"), ("Plans.pdf", "/tmp/data/plans"), ("Column Descriptions.pdf", "/tmp/data/column"),("DataSet.csv", "/tmp/data/csv_columns_combined"),("Column Descriptions.pdf", "/tmp/data/csv_columns_combined")]
    download_files_from_s3(bucket_name, files)
    
ed_time=time.time()
print("Download TIME:", ed_time-st_time)

st_time=time.time()
# classifier = pipeline("zero-shot-classification",
#                       model="facebook/bart-large-mnli")
# classifier = pipeline("zero-shot-classification",
                      # model="bart-large-mnli")
ed_time=time.time()
# print("Classifier TIME:", ed_time-st_time)

# Uncomment to specify your OpenAI API key here, or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""

def get_response_bedrock(query,directory_path,openai_api_key):
    # This example uses text-davinci-003 by default; feel free to change if desired. 
    # Skip openai_api_key argument if you have already set it up in environment variables (Line No: 7)
    # llm_predictor = LLMPredictor(llm=OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4"))
    
    llm = Bedrock(model = "anthropic.claude-3-5-sonnet-20240620-v1:0",
                 region_name="us-east-1",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token)
    embed_model = BedrockEmbedding(model = "amazon.titan-embed-text-v1",
                                  region_name="us-east-1",
                                aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key,
                                aws_session_token=aws_session_token)

    Settings.llm = llm
    Settings.embed_model = embed_model

    sentence_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=20,
)

    splitter = SafeSemanticSplitter(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

    if os.path.isdir(directory_path): 
        # load the documents and create the index
        documents = SimpleDirectoryReader(input_dir=directory_path, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents,transformations=[splitter])

        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        print("FULL RESPONSE>>>>>>>>>>")
        print(dir(response))
        response = response.response
        
        if response is None:
            return "Nothing found"
        else:
            return response
    else:
        return(f"Not a valid directory: {directory_path}")

def get_response(query,directory_path,openai_api_key):
    # This example uses text-davinci-003 by default; feel free to change if desired. 
    # Skip openai_api_key argument if you have already set it up in environment variables (Line No: 7)
    # llm_predictor = LLMPredictor(llm=OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4"))
    
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4")
    llm_predictor = LLMPredictor(llm=llm)
    
    # Configure prompt parameters and initialise helper
    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    if os.path.isdir(directory_path): 
        # Load documents from the 'data' directory
        documents = SimpleDirectoryReader(directory_path).load_data()
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        
        response = index.query(query)
        if response is None:
            return "Nothing found"
        else:
            return response
    else:
        return(f"Not a valid directory: {directory_path}")
def get_benefits_from_query(user_query,sql_queries,openai_key):
    benefits=[]
    # prompt=f"""USER QUERY: {user_query}
    # SQL QUERIES: {sql_queries}
    
    # Given the USER QUERY, and the SQL QUERIES which fetches the relevant plans, Please give {len(sql_queries)} text lines explaining how each of them help the user satisfy their needs but also highligh how each is different from the others.
    # Each SQL QUERY fetches a collection of plans. So when comparing, you are comparing one set of plans to the other {len(sql_queries)-1} plans.

    # Write from the perspective of a sales person who is trying to sell the Plans to the user. Donot mention about the SQL queries to the user, just say here are SOME PLANS, or I have found SOME PLANS or something like that.
    
    # RETURN A PYTHON LIST ONLY NO EXTRA TEXT where each element corresponds to one of the queries.
    # """
    # resp1=get_response(prompt,"./plans",openai_key)
    for sql_query in sql_queries:
        prompt=f"""USER QUERY: {user_query}
        SQL QUERY: {sql_query}
        Given the USER QUERY, and the SQL QUERY which fetches the relevant plans, Please give an explaination to the user how the plans fetched by the QUERY helps satisfy their needs.
        Just say here are SOME PLANS, or I have found SOME PLANS or something like that.
        Write from the perspective of a sales person who is trying to sell the Plans to the user. Donot mention about the SQL query to the user, just say here are SOME PLANS, or I have found SOME PLANS or something like that.
        """
        resp1=get_response(prompt,"/tmp/data/plans",openai_key)
        benefits.append(resp1.response)
    return benefits

def lambda_handler(event, context):
    print("EVENT................")
    print(event)
    body_dict=event
    # body_dict = json.loads(event["body"])
    query=f"""QUERY: {body_dict["query"]}"""
    # query="""QUERY: I have a budget of 50 OMR, do you have any plans with unlimited talktime and atleast 20 GB local data?"""
    q2ask=f"""{query}
    I want to design an sql query to satisfy the QUERY.
    Which Columns do I need to set constraints on to satisfy the QUERY? give a python list."""

    compulory_columns=body_dict["compulory_columns"]
    openai_key=''
    # openai_key=os.environ['OPENAI_API_KEY']
    
    # resp1=get_response(q2ask,"/mnt/lama_index/./column","")
    resp1=get_response_bedrock(q2ask,"/tmp/data/column",openai_key)
    print(resp1)
    # resp1=resp1.response
    for i,letter in enumerate(resp1):
        if(letter=="["):
            start=i
        if(letter=="]"):
            end=i
    resp1 = ast.literal_eval(resp1[start:end+1])
    print("Relevant Columns", resp1)
    
    # prompt=f"""
    # {query}
    
    # Given the QUERY and the list of relevant columns:
    # {resp1}
    
    # Give me some SQL queries to run on DataSet.csv to satisfy the query. I am asking for a list of queries because if one query fails then I can try the next one. Select all columns, only conditions need to be decided.

    # No need for ORDER BY. No need for LIKE %% if QUERY does not ask for it specifically.
    
    # Return a python list.
    # """

    # resp=get_response(prompt,"./csv",openai_key)
    # print("Answer....")
    # print(resp)
    # resp=resp.response
    # for i,letter in enumerate(resp):
    #     if(letter=="["):
    #         start=i
    #     if(letter=="]"):
    #         end=i
    # resp = ast.literal_eval(resp[start:end+1])
    
    # prompt=f"""
    # {query}
    
    # Given the QUERY and the list of relevant columns:
    # {resp1}
    
    # Give me an SQL query to run on /tmp/data/csv/DataSet.csv to satisfy the query. Select all columns, only conditions need to be decided.

    # No need for ORDER BY. No need for LIKE %% if QUERY does not ask for it specifically.
    
    # Return only the SQL query.
    # """
    prompt=f"""
    {query}
    
    Given the QUERY and the list of relevant columns:
    {resp1}
    
    Give me an SQL query to run on /tmp/data/csv/DataSet.csv to satisfy the query. Select all columns, only conditions need to be decided.

    No need for ORDER BY.
    If any categorical value is needed then don't search exactly, the string should be close to the given value.
    
    Return only the SQL query.
    """
    resp=get_response_bedrock(prompt,"/tmp/data/column",openai_key)
    print("Answer....")
    print(resp)
    # resp=resp.response

    # benefits=get_benefits_from_query(query,resp,openai_key)
    # class_result=classifier(body_dict["query"],resp1,multi_label=True)
    url = "https://3gu4uqkxbg.execute-api.ap-south-1.amazonaws.com/default/bart"

    params = {
        "query": body_dict["query"],
        "column": resp1
    }
    
    headers = {
        # 'x-api-key': os.environ['X_API_KEY'],
        'x-api-key': '',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    print("Bart response",response.text)
    class_result=json.loads(response.text)
    class_result=class_result[0]
    print(class_result)
    csv_file_path = '/tmp/data/csv/DataSet.csv'
    df = pd.read_csv(csv_file_path)
    from pandasai.llm import OpenAI  
    # llm = OpenAI(api_token=openai_key,model_name="gpt-4")
    llm = BedrockClaude(bedrock_runtime_client=bedrock_client,model="anthropic.claude-3-5-sonnet-20240620-v1:0")
    # llm = ChatOpenAI(openai_api_key=openai_key, temperature=0, model_name="gpt-4")
    sdf = SmartDataframe(df,config={"llm": llm,"save_logs":False,"enable_cache":False})
    # query=max(resp, key=len)
    # res=sdf.chat(query)
    labels=class_result['labels']
    labels.reverse()
    stored_weights=class_result['scores']
    stored_weights.reverse()
    plans=[]
    final_queries=[]
    # for query in resp:
    DataSet = pd.read_csv(csv_file_path)
    query=resp
    print("QUERY CREATED...",query)

    matpolt_handling = """\n Replace plt.show(...) with plt.show(block=false)"""
    matpolt_handling = """\n If any chart is made don't display it just save it. plt.show(block=false)"""
    # matpolt_handling = ""
    res=sdf.chat(query + matpolt_handling)
    # duckdb.query(query).df()
    print(type(resp1))

    for q_i,column in enumerate(labels):
        remove_prompt=f"""
        SQL Query:{query}
    
        Given The SQL Query, remove the condition about the {column} and return the resulting SQL Query.
        Only return the new Query.
        """
        print(f"current results: {type(res)}")
        # print(res)
        if(type(res)==list):
            res=res[0]

        empty=False
        try:
            if(res.empty):
                empty=True
        except:
            empty=False
            
        if(type(res)==str):
            print(f"removing {column}")
            query=get_response_bedrock(remove_prompt,"/tmp/data/csv_columns_combined",openai_key)
            # query=query.response
            print(f"new query {query}")
      
        elif(empty):
            print(f"removing {column}")
            query=get_response_bedrock(remove_prompt,"/tmp/data/csv_columns_combined",openai_key)
            # query=query.response
            print(f"new query {query}")
            
        else:
            break
            
        res=sdf.chat(query + matpolt_handling)
        # duckdb.query(query).df()
        
    final_queries.append(query)
    # if(type(res)!=str):
    if isinstance(res, pd.DataFrame):
        print(f"Final res...... {query}")
        # print(res)
        print("Priority:",class_result)
        resp1.extend(compulory_columns)
        resp1=list(set(resp1))
        try:
            res=res[resp1]
        except:
            print("Could not get columns subset")
        try:
            res = res.sort_values(by='price')
        except:
            pass
        # sdf = SmartDataframe(res,config={"llm": llm,"save_logs":False,"enable_cache":False})
        # benefits=sdf.chat(body_dict["query"]+matpolt_handling)
        res = res.head(3)
        plans.append(res.to_dict())
    else:
        # benefits=sdf.chat(body_dict["query"]+matpolt_handling)
        plans.append(res)
    benefits=sdf.chat(body_dict["query"]+matpolt_handling)
    # print(res)
    # return {'SQL queries': ["SELECT * FROM DataSet WHERE 'Local Data(GB)' >= 20", "SELECT * FROM DataSet WHERE 'GCC Data(GB)' >= 20", "SELECT * FROM DataSet WHERE 'GCC & Local Combined Data' >= 20"], 'benefits': ['Here are some plans that offer at least 20 GB of data per month:\n- Baqati Bronze: 24 GB combined local and GCC data\n- Baqati Silver: 40 GB local and GCC data\n- Baqati Gold: 60 GB local and GCC data\n- Baqati Platinum: 80 GB local and GCC data\n- Baqati Al Ufuq: 50 GB GCC data\n\nThese plans provide ample data for your communication needs, ensuring that you stay connected without worrying about running out of data. With options ranging from 24 GB to 80 GB, you can choose the plan that best fits your data usage requirements. Feel free to explore these options and select the one that suits you best.', 'Here are some plans that offer at least 20 GB of data per month:\n- Baqati Bronze: 24 GB combined local and GCC data\n- Baqati Silver: 40 GB local and GCC data\n- Baqati Gold: 60 GB local and GCC data\n- Baqati Platinum: 80 GB local and GCC data\n- Baqati Al Ufuq: 50 GB GCC data\n\nThese plans provide ample data for your communication needs, ensuring that you stay connected without worrying about running out of data. With options ranging from 24 GB to 80 GB, you can choose the plan that best suits your data usage requirements. Feel free to explore these options and select the one that fits your needs perfectly.', 'Here are some plans that offer at least 20 GB of data per month: Baqati Bronze, Baqati Silver, Baqati Gold, Baqati Platinum, and Baqati Al Ufuq. These plans provide a generous amount of data to cater to your high data usage needs. With these plans, you can enjoy seamless browsing, streaming, and downloading without worrying about running out of data. Additionally, you can also explore our international calling options with different peak and off-peak rates to stay connected with your loved ones abroad. Feel free to explore these options and choose the one that best suits your requirements.'], 'plans': [{'Pack Type': {3: 'Regular', 4: 'Regular', 5: 'Regular', 6: 'Regular', 7: 'Regular', 8: 'Regular', 9: 'Regular', 10: 'Regular', 11: 'Regular', 12: 'Regular', 13: 'Data Top up'}, 'Plan Name': {3: 'Baqati Bronze', 4: 'Baqati Silver', 5: 'Baqati Gold', 6: 'Baqati Platinum', 7: 'Baqati Al Ufuq', 8: 'Baqati Bronze', 9: 'Baqati Silver', 10: 'Baqati Gold', 11: 'Baqati Platinum', 12: 'Baqati Al Ufuq', 13: 'Add On  Data '}, 'Local Data(GB)': {3: 24, 4: 40, 5: 60, 6: 80, 7: 99999999, 8: 24, 9: 40, 10: 60, 11: 80, 12: 99999999, 13: 99999999}, 'GCC Data(GB)': {3: 24, 4: 40, 5: 60, 6: 80, 7: 50, 8: 24, 9: 40, 10: 60, 11: 80, 12: 50, 13: 0}, 'GCC & Local Combined Data': {3: 'Yes', 4: 'Yes', 5: 'Yes', 6: 'Yes', 7: 'No', 8: 'yes', 9: 'yes', 10: 'yes', 11: 'yes', 12: 'No', 13: 'No'}}, {'Pack Type': {3: 'Regular', 4: 'Regular', 5: 'Regular', 6: 'Regular', 7: 'Regular', 8: 'Regular', 9: 'Regular', 10: 'Regular', 11: 'Regular', 12: 'Regular'}, 'Plan Name': {3: 'Baqati Bronze', 4: 'Baqati Silver', 5: 'Baqati Gold', 6: 'Baqati Platinum', 7: 'Baqati Al Ufuq', 8: 'Baqati Bronze', 9: 'Baqati Silver', 10: 'Baqati Gold', 11: 'Baqati Platinum', 12: 'Baqati Al Ufuq'}, 'Local Data(GB)': {3: 24, 4: 40, 5: 60, 6: 80, 7: 99999999, 8: 24, 9: 40, 10: 60, 11: 80, 12: 99999999}, 'GCC Data(GB)': {3: 24, 4: 40, 5: 60, 6: 80, 7: 50, 8: 24, 9: 40, 10: 60, 11: 80, 12: 50}, 'GCC & Local Combined Data': {3: 'Yes', 4: 'Yes', 5: 'Yes', 6: 'Yes', 7: 'No', 8: 'yes', 9: 'yes', 10: 'yes', 11: 'yes', 12: 'No'}}, {'Pack Type': {1: 'Regular', 2: 'Regular', 3: 'Regular', 4: 'Regular', 5: 'Regular', 6: 'Regular'}, 'Plan Name': {1: 'Baqati Blue', 2: 'Baqati Blue Plus', 3: 'Baqati Bronze', 4: 'Baqati Silver', 5: 'Baqati Gold', 6: 'Baqati Platinum'}, 'Local Data(GB)': {1: 14, 2: 16, 3: 24, 4: 40, 5: 60, 6: 80}, 'GCC Data(GB)': {1: 14, 2: 16, 3: 24, 4: 40, 5: 60, 6: 80}, 'GCC & Local Combined Data': {1: 'Yes', 2: 'Yes', 3: 'Yes', 4: 'Yes', 5: 'Yes', 6: 'Yes'}}]}

    # for plan in plans:
    #     print(pd.DataFrame(plan))

    # benefits=get_benefits_from_query(query,final_queries,openai_key)
    print("Got Benefits ...............")
    # # print({"SQL queries":resp,"labels":labels,"stored_weights":stored_weights,"Final Queries":final_queries,"plans":plans})
    # # return {"SQL queries":resp,"benefits":benefits,"labels":labels,"stored_weights":stored_weights,"Final Queries":final_queries,"plans":plans}
    # to_ret={"SQL queries":resp,"labels":labels,"stored_weights":stored_weights,"Final Queries":final_queries,"plans":plans}
    # return {"statusCode": 200,"headers": {"Content-Type": "application/json"},"body": json.dumps(to_ret)}


    try:
        if(type(benefits)!=str):
            benefits=None
        to_ret = {
            "User query":body_dict["query"],
            "SQL queries": resp,
            "labels": labels,
            "stored_weights": stored_weights,
            "Final Queries": final_queries,
            "plans": plans,
            "benefits":benefits
        }
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(to_ret)
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    
    # return {"SQL queries":resp,"labels":labels,"stored_weights":stored_weights,"Final Queries":final_queries,"plans":plans}

# def lambda_handler2(event, context):
#     openai_key=""
#     # openai_key=os.environ['OPENAI_API_KEY']
#     compulory_columns=event["compulory_columns"]
#     relevant_columns=event["relevant_columns"]
#     stored_weights=event["stored_weights"]

#     query=f"""QUERY: {event["request"]}"""
#     # query="""QUERY: I have a budget of 50 OMR, do you have any plans with unlimited talktime and atleast 20 GB local data?"""
#     q2ask=f"""{query}
#     I want to design an sql query to satisfy the QUERY.
#     Which Columns do I need to set constraints on to satisfy the QUERY? give a python list."""

#     resp1=get_response(q2ask,"./column",openai_key)
#     print(resp1)
#     resp1=resp1.response
#     for i,letter in enumerate(resp1):
#         if(letter=="["):
#             start=i
#         if(letter=="]"):
#             end=i
#     resp1 = ast.literal_eval(resp1[start:end+1])

    
#     dummy={"New Query": "The new SQL Query","Message":"The message thanking the user for clarification and telling that how the new plans will satisfy the user's request in one line."}
#     prompt = f""" SQL Query: {event["query"]} 

#     User Request: {event["request"]}

#     Given the User Request, modify the SQL Query and return the New Query along with a Message saying thankyou for the clarification...
#     complete the rest of the Message appropriately.
    
#     Return a JSON of the format:
#     {dummy}

#     Return only the JSON, no extra generation. No extra Spaces or tabs. Use double quotes to enclose the keys and values.
#     """
#     resp=get_response(prompt,"./csv",openai_key)
#     resp=resp.response
#     for i,letter in enumerate(resp):
#         if(letter=="{"):
#             start=i
#         if(letter=="}"):
#             end=i
#     print("Modified resp....")
#     resp=resp.replace("\n","")
#     resp=resp.replace("\t","")
#     print(resp)
#     resp = ast.literal_eval(resp[start:end+1])


#     plans=[]
#     final_queries=[]
#     query=resp["New Query"]
#     benefits=resp["Message"]
#     class_result=classifier(event["request"],resp1,multi_label=True)
#     print(class_result)

#     csv_file_path = './csv/DataSet.csv'
#     df = pd.read_csv(csv_file_path)
#     from pandasai.llm import OpenAI
#     llm = OpenAI(api_token=openai_key,model_name="gpt-4")
#     sdf = SmartDataframe(df,  {"enable_cache": False}, config={"llm": llm})
#     # query=max(resp, key=len)
#     # res=sdf.chat(query)
#     labels=class_result['labels']
#     labels.reverse()
#     current_weights=class_result['scores']
#     current_weights.reverse()

#     set1=set(labels)
#     set2=set(relevant_columns)
#     set_union= set1.union(set2)

#     final_labels=[]
#     final_weights=[]

#     for label in set_union:
#         final_labels.append(label)
#         if(label in labels):
#             w1=current_weights[labels.index(label)]
#         else:
#             w1=0
#         if(label in relevant_columns):
#             w2=stored_weights[relevant_columns.index(label)]
#         else:
#             w2=0
#         final_weights.append(w1+w2)

#     sorted_labels = [label for _, label in sorted(zip(final_weights, final_labels))]
#     final_weights.sort()
    
#     res=sdf.chat(query)
#     print(type(resp1))
#     memory_index=None
#     memory_query=None
#     for q_i,column in enumerate(sorted_labels):
#         if(((final_weights[q_i])/(event["chat_length"]+1))>=0.2):
#             if(memory_index==None):
#                 memory_index=q_i
#                 memory_query=query
#         remove_prompt=f"""
#         SQL Query:{query}
    
#         Given The SQL Query, remove the condition about the {column} and return the resulting SQL Query.
#         Only return the new Query.
#         """
#         print(f"current results: {type(res)}")
#         print(res)
#         if(type(res)==list):
#             res=res[0]
            
#         if(type(res)==str):
#             print(f"removing {column}")
#             query=get_response(remove_prompt,"./column",openai_key)
#             query=query.response
#             print(f"new query {query}")
            
#         elif(res.empty):
#             print(f"removing {column}")
#             query=get_response(remove_prompt,"./column",openai_key)
#             query=query.response
#             print(f"new query {query}")
            
#         else:
#             break
            
#         res=sdf.chat(query)
#     final_queries.append(query)
#     resp1=sorted_labels[0:]
#     if(type(res)!=str):
#         print(f"Final res...... {query}")
#         print(res)
#         print("Priority:",class_result)
#         resp1.extend(compulory_columns)
#         resp1=list(set(resp1))
#         res=res[resp1]
#         plans.append(res.to_dict())
#     else:
#         plans.append(res)
#     print(res)
    
#     if(memory_index==None):
#         print("No memory")
#         return {"SQL queries":resp["New Query"],"benefits":benefits,"labels":sorted_labels,"stored_weights":final_weights,"Final Queries":final_queries,"plans":plans,"chat_length":event["chat_length"]+1}
#     else:
#         print("Memory", memory_index)
#         return {"SQL queries":memory_query,"benefits":benefits,"labels":sorted_labels[memory_index:],"stored_weights":final_weights[memory_index:],"Final Queries":final_queries,"plans":plans,"chat_length":event["chat_length"]+1}

# if(__name__=="__main__"):
#     # print(lambda_handler({"query":"My GCC calling minutes are exhausted on my post paid plan. I need atleast 30 minutes of talk time for GCC. What are my options?","compulory_columns":["Plan Name","Cost(OMR)"]}, 0))
#     query_0="I need a fully furnished 3 BHK appartment near Dubai Marina."
#     query_1="""Which neighbourhood has the maximum availabilities ?"""
#     query_2="How many Studio apartments are available ? "
#     query_3="In which area you find the cheapest studio apartment ? "
#     query_4="Whats the average price of 1 BHK apartment in downtown? "
#     query_5="If I want to stay close to Palm Jumeirah what should be my budget for buying a 2 BHK apartment ? "
#     query_6="Rank the property with cheapest to most expensive to buy ."
#     query_7="I need a maximum of 1200 sqft to live . List all the properties which have a gym and a swimming pool that meet my requirements"
#     query_8="Is quality of property driving the price of the property ?"
#     query_9="Which not very posh locality has maximum appartments with servant quarters?"
#     print(lambda_handler({"query":query_9,"compulory_columns":["id","price"]}, 0))

# download_file_from_s3(bucket_name, file_name, destination_folder)

# file_name = "DataSet.csv"
# destination_folder = "/tmp/data/csv"

# download_file_from_s3(bucket_name, file_name, destination_folder)

# file_name = "Plans.pdf"
# destination_folder = "/tmp/data/plans"

# download_file_from_s3(bucket_name, file_name, destination_folder)
# print(lambda_handler2({"request":"None of these plans have Local minutes I need Local minutes of 1.","query":"SELECT * FROM DataSet WHERE 'International Minutes' >= 180 AND 'GCC & Local Combined Data' >= 1","compulory_columns":["Plan Name","Cost(OMR)"],"relevant_columns":['Local Minutes', 'Local Data(GB)', 'GCC Data(GB)', 'GCC & Local Combined Data', 'International Minutes'],"stored_weights":[0.004888167139142752, 0.012974861077964306, 0.11856455355882645, 0.21607129275798798, 0.3088933527469635],"chat_length":1}, 0))