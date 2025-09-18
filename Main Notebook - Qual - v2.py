# Databricks notebook source
!pip install uv
!pip install -U "huggingface_hub[cli]"

dbutils.library.restartPython()

# COMMAND ----------

!uv pip install transformers==4.41.0 sentence-transformers==4.1.0 --native-tls
dbutils.library.restartPython()

# COMMAND ----------

!uv pip install transformers>=4.52.0 torch>=2.6.0 peft>=0.15.2 torchvision pillow sentence-transformers==4.1.0 flash-attn --no-build-isolation --native-tls
dbutils.library.restartPython()

# COMMAND ----------

# this is for jina v4
# !uv pip install transformers>=4.55.3 torch>=2.8.0 peft>=0.17.1 torchvision pillow sentence-transformers==4.1.0 flash-attn==2.8.3 --no-build-isolation --native-tls
# dbutils.library.restartPython()

# COMMAND ----------

!uv pip install --upgrade scikit-learn numpy scipy pandas pyarrow --native-tls
dbutils.library.restartPython()

# COMMAND ----------

# this is for gpt-oss
!uv pip install --pre vllm==0.10.1+gptoss --extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match --native-tls

dbutils.library.restartPython()

# COMMAND ----------

!uv pip install --force-reinstall triton==3.4.0 --native-tls
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Initialization

# COMMAND ----------

import os
import traceback
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# from transformers import RobertaConfig
import torch.nn as nn
# from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import pickle
from pathlib import Path
import torch
import torch.nn.functional as F
import evaluate

# from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

from scipy.stats import pearsonr
from datasets import Dataset, DatasetDict, load_dataset
# from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from huggingface_hub import snapshot_download, login
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     DataCollatorWithPadding
# )
from shutil import copyfile
import os, re
import pandas as pd
import numpy as np
import requests
import json
# from transformers import pipeline
# import mlflow
# from mlflow.models import infer_signature
# from mlflow.transformers import generate_signature_output
# from mlflow.tracking import MlflowClient
# from mlflow.pyfunc import PythonModelContext
# import cloudpickle
# from transformers import BitsAndBytesConfig, LlamaForSequenceClassification
from vllm import LLM, SamplingParams
import vllm

import os
import re
import json
from tqdm import tqdm
from docx import Document
from docx.document import Document as _Document 
from docx.table import Table 
from docx.text.paragraph import Paragraph
import pandas as pd

simlilarity_matching = False

vllm.__version__

# COMMAND ----------

from pathlib import Path
import shutil
import os, re
import os
default_n_threads = 32
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import pandas as pd
import numpy as np
from huggingface_hub import snapshot_download, login



# COMMAND ----------

input_folder = Path("/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/NST/Eaglet/" )     
output_folder = Path("/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/NST/Eaglet/output_quals_v2" )
input_folder.mkdir(exist_ok=True, parents=True)
output_folder.mkdir(exist_ok=True, parents=True)

models = {
    "mistralai--Mistral-7B-Instruct-v0.2": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "revision": "41b61a33a2483885c981aa79e0df6b32407ed873"
    },
    "mistralai--Mistral-7B-Instruct-v0.3": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
    },
    "neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w4a16": {
        "model_id": "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16",
        "revision": "8c670bcdb23f58a977e1440354beb7c3e455961d"
    },
    "meta-llama--Llama-3.1-8B-Instruct": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659"
    },
    "neuralmagic--Meta-Llama-3.1-70B-Instruct-FP8": {
        "model_id": "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
        "revision": "08b31c0f951f2227f6cdbc088cdb6fd139aecf0f"
    },
    "microsoft--Phi-4-mini-instruct": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "revision": "c0fb9e74abda11b496b7907a9c6c9009a7a0488f"
    },
    "cortecs--Llama-3.3-70B-Instruct-FP8-Dynamic": {
        "model_id": "cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic",
        "revision": "3722358cc2b990b22304489b2f87ef3bb876d6f6"
    },
    "gpt-oss-120b": {
        "model_id": "/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/Models/gpt-oss-120b",
        "revision": None
    },
    "jinaai--jina-embeddings-v4": {
        "model_id": "jinaai/jina-embeddings-v4",
        "revision": "737fa5c46f0262ceba4a462ffa1c5bcf01da416f"
    },
    "Qwen3-Embedding-8B": {
        "model_id": None,
        "revision": None
    }
}

# COMMAND ----------

# %sh cd /Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/Models/
# git lfs install
# git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2

# COMMAND ----------

# %sh ls /root/.cache/huggingface/hub/models--cortecs--Llama-3.3-70B-Instruct-FP8-Dynamic/snapshots

# COMMAND ----------

# %sh cp -L -r '/root/.cache/huggingface/hub/models--cortecs--Llama-3.3-70B-Instruct-FP8-Dynamic' '/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/Models/'

# COMMAND ----------

from pathlib import Path

def download_model(model_id):
    # download the model form HF just once
    from huggingface_hub import snapshot_download, login
    login(token='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')
    # model_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"
    # revision = "cf47bb3e18fe41a5351bc36eef76e9c900847c89"
    snapshot_location = snapshot_download(repo_id=model_id)

def get_snapshot_loc(active_model, copy_ = True):
    model_id = models[active_model]['model_id']
    revision = models[active_model]['revision']
    snapshot_location = f"/root/.cache/huggingface/hub/{active_model}"
    if revision:
        try:
            shutil.copytree(f"/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/Models/models--{active_model}", 
                            f"/root/.cache/huggingface/hub/models--{active_model}")
        except FileExistsError:
            print(f"Directory exists...")
            pass
        snapshot_location = snapshot_download(repo_id=model_id, 
                                              revision=revision, 
                                              cache_dir="/root/.cache/huggingface/hub")
    else:
        if copy_:
            try:
                shutil.copytree(f"/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/Models/{active_model}", 
                                f"/root/.cache/huggingface/hub/{active_model}")
            except FileExistsError:
                print(f"Directory exists...")
                pass
            snapshot_location = f"/root/.cache/huggingface/hub/{active_model}"
        else:
            snapshot_location = model_id
    return snapshot_location

def instruction_format(sys_message: str, query: str, template=''):
    if template == 'Phi':
        return f'<|system|> {sys_message} <|end|><|user|> {query} <|end|><|assistant|>'
    
    if template == 'Llama':
        return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    
    if template == "GPT":
        return f'''<|start|>system<|message|>{sys_message}\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{query}<|end|><|start|>assistant'''
    # note, don't "</s>" to the end
    return f'<s> [INST] {sys_message} [/INST]\nUser: {query}\nAssistant: '

def get_vllm_model(snapshot_location, number_gpus = 1, max_model_len = 512):
    
    llm = LLM(model=snapshot_location, 
              tensor_parallel_size=number_gpus, 
              max_model_len=max_model_len)#, distributed_executor_backend='ray')
    return llm

def extract_entity_from_text(df_, input_col, entity_type, input_type, llm):
    sys_msg = f'''You are a helpful super intellignet AI assistant. Extract the {entity_type} from {input_type} in the input. Report the main {entity_type} in a simple format. Do not report other information. If you can't find the {entity_type} output "None".'''

    list_ = df_[input_col].values.tolist()
    print(f"Number of rows: {len(list_)}")
    full_prompts = [instruction_format(sys_msg, t, template='Phi') for t in list_]

    sampling_params = SamplingParams(
                                    max_tokens=1024,
                                    temperature=0.0,
                                    top_p=1, 
                                    frequency_penalty=0, 
                                    presence_penalty=0, 
                                    n=1, 
                                    best_of=1
                                    )
    assert len(list_) == len(full_prompts), f"Error in lens: {len(list_)} != {len(full_prompts)}"
    outputs = llm.generate(full_prompts, sampling_params=sampling_params)
    responses = [output.text.strip() for out in outputs for output in out.outputs]
    df_[f'Extracted {entity_type}'] = responses
    return df_

def get_sim_between_cols(df_, input_pair_cols, entity_type, llm):
    sys_msg = f'''You are a helpful super intellignet AI assistant. output a similarity score between the pair of {entity_type} in the input list. Note the score must be between 0 and 1. Do not generate any text, output the score only.'''
    
    list_ = df_[input_pair_cols].values.tolist()
    full_prompts = [instruction_format(sys_msg, "['" + "', '".join(t) + "']") for t in list_]
    sampling_params = SamplingParams(temperature=0, 
                                 top_p=1, 
                                 frequency_penalty=0, 
                                 presence_penalty=0, 
                                 n=1, 
                                 best_of=1, 
                                 max_tokens=1024)
    outputs = llm.generate(full_prompts, sampling_params=sampling_params)
    responses = [[output.text.strip()] for out in outputs for output in out.outputs]
    
    df_out = df_
    scores = [float(re.findall(r"\d\.\d+", item)[0]) for l in responses for item in l]
    df_out['score'] = scores
    # df_out.drop_duplicates(subset=['ANZSCO Title', 'ANZSCO Code', 'Column2.2'], inplace=True)
    return df_out

def extarct_edu_level(df_, input_col, llm, max_token=2048):
    sys_msg = f'''You are a helpful super intellignet AI assistant. Extract the "Education level" followed by "||" and the EXACT "Education field or Major" from the input text. DO NOT report duplicate education level and field. Just report distinct the education level and education field separated by "\n", DO NOT generate any extra sentences. Make sure the output is consistant with the output format'''

    # sys_msg = f'''You are a helpful super intellignet AI assistant. Extract the sets of education level followed by education field separated by "||" from the input text. Just report distinct set separated by "\n". DO NOT report duplicate set of education level and field and DO NOT generate any extra sentences. '''

    list_ = df_[input_col].values.tolist()
    print(f"Number of rows: {len(list_)}")
    full_prompts = [instruction_format(sys_msg, t, template='Llama') for t in list_]

    sampling_params = SamplingParams(
                                    max_tokens=max_token,
                                    temperature=0.0,
                                    top_p=1, 
                                    frequency_penalty=0,
                                    presence_penalty=0, 
                                    n=1, 
                                    best_of=1
                                    )
    assert len(list_) == len(full_prompts), f"Error in lens: {len(list_)} != {len(full_prompts)}"
    outputs = llm.generate(full_prompts, sampling_params=sampling_params)
    responses = [output.text.strip() for out in outputs for output in out.outputs]
    return responses

def write_to_db(sdf_s, db_name):
    spark.sql(f'DROP TABLE IF EXISTS {db_name}')
    sdf_s.write.saveAsTable(f'{db_name}')

def write_output_to_excel(dict_df: dict, output_folder, file_name, post_fix='scored'):
    from shutil import copyfile
    dbfs_file = output_folder / f"{file_name}_{post_fix}.xlsx"
    local_file = './temp.xlsx'
    # df_out.to_excel(local_file, index=False)
    with pd.ExcelWriter(local_file) as writer:
        for sheet in dict_df:
            dict_df[sheet].to_excel(writer, sheet_name=sheet, index=False)
    copyfile(local_file, dbfs_file)


def extarct_(df_, input_col, llm, sys_msg=None, max_token=2048, cols=[]):   

    # sys_msg = f'''You are a helpful super intellignet AI assistant. Extract the sets of education level followed by education field separated by "||" from the input text. Just report distinct set separated by "\n". DO NOT report duplicate set of education level and field and DO NOT generate any extra sentences. '''
    if sys_msg:
        list_ = df_[input_col].values.tolist()
        print(f"Number of rows: {len(list_)}")

        full_prompts = [instruction_format(sys_msg, t, template='Llama') for t in list_]
    else:
        full_prompts = [instruction_format(get_sys_msg(qual, occp), t, template='Llama') for t, qual, occp in df_[[input_col] + cols].values.tolist()]

    sampling_params = SamplingParams(
                                    max_tokens=max_token,
                                    temperature=0.0,
                                    top_p=1, 
                                    frequency_penalty=0,
                                    presence_penalty=0, 
                                    n=1, 
                                    best_of=1
                                    )
    assert df_.shape[0] == len(full_prompts), f"Error in lens: {df_.shape[0]} != {len(full_prompts)}"
    outputs = llm.generate(full_prompts, sampling_params=sampling_params)
    responses = [output.text.strip() for out in outputs for output in out.outputs]
    return responses

def deduplicate_text(texts: list, model, thr):
    # Generate embeddings for the texts
    task = "text-matching"
    embeddings = model.encode(
        texts,
        task=task,
        prompt_name=task,
        show_progress_bar=True
    )

    # Calculate cosine similarity between all pairs of embeddings
    cosine_scores = cos_sim(embeddings, embeddings).numpy()

    deduplicated_texts = []
    processed_indices = set()

    # Iterate through the similarity matrix to identify and remove duplicates
    for i in range(len(texts)):
        if i not in processed_indices:
            deduplicated_texts.append(texts[i])
            processed_indices.add(i)
            for j in range(i + 1, len(texts)):
                if cosine_scores[i][j] > thr:  # Adjust threshold as needed
                    processed_indices.add(j)
    return deduplicated_texts

def get_similarty_by_embedding(df_x, df_y, x_col, y_col, models):
    df_x = df_x[[x_col]]
    df_y = df_y[[y_col]]

    df_x = df_x.dropna(subset=[x_col])
    df_x.reset_index(inplace=True)
    df_x.rename({'index': 'actual_index_x'}, axis='columns', inplace=True)
    df_y = df_y.dropna(subset=[y_col])
    df_y.reset_index(inplace=True)
    df_y.rename({'index': 'actual_index_y'}, axis='columns', inplace=True)

    df_y.reset_index(inplace=True)
    df_x.reset_index(inplace=True)
    id_to_label = {id_: label for id_, label in df_y[['index', y_col]].values.tolist()}
    query = df_y[y_col].values.tolist()
    docs = df_x[x_col].values.tolist()
    # task = "text-matching"
    # # task = "retrieval"
    # query_embedding = model.encode(
    #                         query,
    #                         task=task,
    #                         prompt_name='passage',
    #                         batch_size=64,
    #                         show_progress_bar=False,
    #                         convert_to_numpy=True,
    #                         normalize_embeddings=True  # L2 normalization for better clustering
    #                         )

    # docs_embeddings = model.encode(
    #                         docs,
    #                         task=task,
    #                         prompt_name='passage',
    #                         batch_size=64,
    #                         show_progress_bar=False,
    #                         convert_to_numpy=True,
    #                         normalize_embeddings=True  # L2 normalization for better clustering
    #                         )
    similarities = None
    coef = 1/len(models)
    for model, kwargs in models:
        query_embedding = model.encode(query, **kwargs)
        docs_embeddings = model.encode(docs, **kwargs)
        if similarities is None:
            similarities = coef * cos_sim(docs_embeddings, query_embedding).numpy()
        else:
            similarities += coef * cos_sim(docs_embeddings, query_embedding).numpy()

    df_y.drop('index', axis='columns', inplace=True)
    df_preds = pd.DataFrame([[id_to_label[id_]] for id_ in similarities.argmax(axis=1)], 
                        columns=['Pred']).reset_index()
    df_sim = pd.DataFrame([[sim] for sim in similarities.max(axis=1)], 
                            columns=['Sim']).reset_index()
    df_l = df_x.merge(df_preds, on='index').merge(df_sim, on='index').drop('index', axis='columns')
    df_l = df_l.merge(df_y, left_on='Pred', right_on=y_col).drop('Pred', axis='columns')
    return df_l

import fitz
import pymupdf
def get_all_text_pdf(fname):
    with pymupdf.open(fname) as doc:  # open document
        text = "\n\n".join([page.get_text() for page in doc])
    return text

def parse_pdf_doc(fname: str) -> dict:
    doc = fitz.open(fname)
    interested_sections = ['Course Name', 'Description', 'Learning Outcomes', 'Subject Content']
    text_dict = {"Course Name": "", "text": ""}
    current_section = None
    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block["type"] == 0: # Text block
                line_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Example: Identify large, bold text as potential heading
                        if span["flags"] == 20 and text_dict['Course Name'] != "":
                            if span['text'] in interested_sections:
                                text_dict['text'] += "\n\n" + span['text'] + ":\n"
                            current_section = span['text']
                        else:
                            line_text += span['text'] + " "
                
                if text_dict['Course Name'] == "" and span['flags'] == 20:
                    text_dict['Course Name'] = line_text.strip()
                    text_dict['text'] += "Course Name: \n" + line_text.strip()
                else:
                    if current_section in interested_sections:
                        text_dict['text'] += line_text
                        
    doc.close()    
    return text_dict

def iter_block_items(parent):
    parent_element = parent.element.body
    for child in parent_element:
        if child.tag.endswith('}p'):
            yield Paragraph(child, parent)
        elif child.tag.endswith('}tbl'):
            yield Table(child, parent)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def parse_docx_doc(fname: str) -> dict:
    doc = Document(fname)

    unit_code, unit_title = None, None
    section = None
    source_text = None
    context = None

    interested_sections = ['Application', 'Elements and Performance Criteria', 
                            'Foundation Skills', 'Performance Evidence', 'Knowledge Evidence']
    
    main_text = ""
    appl_first = None
    for block in iter_block_items(doc):
        dict_ = {}
        if isinstance(block, Paragraph):
            text = block.text
            if not unit_code and re.match(r'^[A-Z]{6}\d{3}', text):
                unit_code = text.split()[0]
                unit_title = text.replace(unit_code, '').strip()
            # print(block.style.name, block.text)
            if 'Heading' in block.style.name:
                main_text += "\n\n" + block.text + "\n"
            else:
                main_text += block.text
        if isinstance(block, Table):
            for row in block.rows:
                main_text += '\n'
                for cell in row.cells:
                    # print(cell.text)
                    main_text += '| ' + cell.text + ' '
    dict_['Unit_Code'] = unit_code
    dict_['Unit_Title'] = unit_title
    dict_['text'] = main_text
    return main_text

# COMMAND ----------

if simlilarity_matching:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    model_sim = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
else:
    active_models = ['gpt-oss-120b']
    llm = get_vllm_model(get_snapshot_loc(active_models[0], False), number_gpus=2, max_model_len=8048)

# COMMAND ----------

df_m = pd.read_excel(input_folder / "Metadata.xlsx").drop_duplicates(["OSCA", "UOC Code"])
uoc_list = df_m['UOC Code'].values.tolist()
display(df_m)
df_m.shape

# COMMAND ----------

doc_path = [str(file) for file in input_folder.glob('**/*.docx') if 'Complete_R' in str(file)]

doc_path = [{'UOC Code':uoc, 'file': f, 'fname': Path(f).name} for f in doc_path for uoc in uoc_list if uoc in f]

df_f = pd.DataFrame(doc_path).drop_duplicates(['UOC Code', 'fname'])
df_mf = df_m.merge(df_f, on='UOC Code', how='left')
display(df_mf), print(len(doc_path))

# COMMAND ----------

from tqdm import tqdm
doc_s = df_mf.to_dict(orient='records')
all_data = []
for doc in tqdm(doc_s):
    doc['text'] = parse_docx_doc(doc['file'])
    all_data.append(doc)
len(all_data)

# COMMAND ----------

all_data[0]['text']

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Engineering

# COMMAND ----------

prompt_v = 'C_1'
sys_msg = """
Extract explicit and implicit skills from university subject outlines (e.g., ACCT2001, LAWS1001), with emphasis on cognitive functions, contextual adaptability, and dynamic evolution. Return the skills in a structured JSON format.

Instructions:
Review the Content, Focus on the following sections:
- Learning Outcomes
- Subject Description
- Subject Content

Skill Definition and Focus:
A "skill" is a valued and purposeful human ability that is acquired through learning and practice. It includes cognitive functions, evolves with experience,  and adapts to different contexts. It is a dynamic function of an individual’s knowledge, experience, and personal attributes.
 
Extraction Tasks:
----------------------------------------------------------------
Task 1: Extract Skills
Identify: Find individual skills explicitly described or implicitly indicated in the source text.
 
Criteria:
Measurable and observable abilities, tasks, or behaviors.
Captures cognitive skills (e.g., decision-making, problem-solving).
Written in action-oriented language (e.g., “Use culturally safe communication,” “Schedule appointments”).
Specific for assessment, yet adaptable to various roles and contexts.
Reflect dynamic nature and continuous learning.
-----------------------------------------------------------------
Task 2: Tag Each Skill with its Noun-based Version
Example: For "Assess individual needs," the noun-based version is "Needs assessment."
Considerations for Cognitive Skills:
 
Incorporate elements like decision-making, judgment, analysis, and problem-solving.
Ensure these skills are captured as they are essential to performing tasks effectively.
-----------------------------------------------------------------
Output Requirements:
Format: Provide results in a JSON array with this structure:
[
 {
    "skill": "string",
    "noun_based_skill": "string"
 }
]
Validity: Keep the JSON strictly valid — suitable for direct parsing

Additional Guidance:
- Emphasize Adaptability: Assess how skills apply across different scenarios and environments.
- Dynamic Evolution: Recognize how skills improve with learning and experience.
- Human-Centric Abilities: Focus on human-specific capabilities, distinct from automation and task.
- Feedback and Iteration: Continuously refine skills understanding based on feedback.
- Keep the JSON strictly valid — suitable for direct parsing

Note: only output the JSON, do not generate any extra sentences.

"""

# COMMAND ----------

prompt_v = '1'
sys_msg = """
You are an expert in job description analysis. Your task is to Extract explicit and implicit skills from the given job description with emphasis on cognitive skills, contextual adaptability, and dynamic nature. Return these skills in a structured JSON format.
 
Extraction Tasks:
Task 1: Extract Skills
Identify: Find individual skills according to the five rules below. Be decisive, but conservative—prefer review when ambiguous.
Skill rules:
R1 (human ability): The item must represent a human ability, not a tool, software, credential, organization, standard, degree, product, or document.
R2 (learnable): It can be acquired or improved through learning and practice.
R3 (applied ability): It involves applying knowledge, experience, and personal attributes (cognitive, interpersonal, psychomotor).
R4 (dynamic/contextual): It is shaped by context, interaction with others, or environmental demands (not a static fact).
R5 (purpose/value): It is purpose-driven and considered valuable in work or life.
 
Mapping guidance:
If the candidate skill is a tool/tech (fails R1) but there is a clear human ability implied by its use, propose a mapped_skill (e.g., “Excel” → “spreadsheet data analysis”; “Forklift” → “forklift operation”). Only map to a concise, action-oriented skill noun phrase.
 
Task 2: Output Requirements:
Format: Provide results in a JSON array with this structure:
[
  {
    "skill": "<extracted skill>",
    "normalized": "<lemmatized short form>",
    "mapped_skill": "<if applicable or null>"
  }
]
Validity: Ensure valid JSON for direct parsing.
 
Additional Guidance:
- Keep rationale short and non-speculative.
- Keep the JSON strictly valid — suitable for direct parsing

Note: only output the JSON, do not generate any extra sentences.

"""

# COMMAND ----------

prompt_v = '2'
sys_msg = """
You are an expert in job description analysis. Your task is to Extract explicit and implicit skills from the given job description with emphasis on cognitive skills, contextual adaptability, and dynamic nature. Return these skills in a structured JSON format.
 
Instructions:
---------------------------------------------------------------------
Review the job description:
 
Skill Definition and Focus:
A "skill" is a valued and purposeful human ability that is acquired through learning and practice. It includes cognitive functions, evolves with experience,  and adapts to different contexts. It is a dynamic function of an individual’s knowledge, experience, and personal attributes.
 
Extraction Tasks:
Task 1: Extract Skills
Identify: Find individual skills explicitly described or implicitly indicated in the job description.
Criteria:
Measurable and observable abilities, tasks, or behaviors.
Captures cognitive skills (e.g., decision-making, problem-solving).
Written in action-oriented language (e.g., “Use culturally safe communication,” “Schedule appointments”).
Specific for assessment, yet adaptable to various roles and contexts.
Reflect dynamic nature and continuous learning.
 
Task 2: Tag Each Skill with its Noun-based Version
Example: For "Assess individual needs," the noun-based version is "Needs assessment."
Considerations for Cognitive Skills:
Incorporate elements like decision-making, judgment, analysis, and problem-solving.
Ensure these skills are captured as they are essential to performing tasks effectively.

Output Requirements:
Format: Provide results in a JSON array with this structure:
[
  {
    "skill": "string",
    "noun_based_skill": "string"
  }
]
Validity: Ensure valid JSON for direct parsing.
 
Additional Guidance:
- Emphasize Adaptability: Assess how skills apply across different scenarios and environments.
- Dynamic Evolution: Recognize how skills improve with learning and experience.
- Human-Centric Abilities: Focus on human-specific capabilities, distinct from automation and task.
- Feedback and Iteration: Continuously refine skills understanding based on feedback.
- Keep the JSON strictly valid — suitable for direct parsing

Note: only output the JSON, do not generate any extra sentences.

"""

# COMMAND ----------

prompt_v = '3'
sys_msg = """
You are an expert in job description analysis and skill extraction. Given a job description, extract individual **skills**, making sure they are generalizable **capabilities** — not task descriptions and return them in structured JSON format for integration into a skill database.

IMPORTANT: Do NOT extract tasks, activities, or instructions. Focus on the **underlying skill or ability** demonstrated by the job description. If the candidate skill is a tool/tech but there is a clear human ability implied by its use, propose a skill (e.g., “Excel” → “spreadsheet data analysis”; “Forklift” → “forklift operation”). Only map to a concise, action-oriented skill noun phrase.

For each skill, output the following fields in JSON format:
Return your answer in strict JSON format as shown below:
[
  {
    "skill": "<The abstracted skill name + context — not a task>",
    "noun_based_skill": "<tag skill with its Noun-based Version>",
    "normalized": "<lemmatized short form>"
  }
]

Additional guidance:
- Avoid skills that are just keywords (e.g., “communication”) without context. Add modifiers: “Culturally appropriate communication”.
- Keep the JSON strictly valid — suitable for direct parsing.

Note: only output the JSON, do not generate any extra sentences.

"""
# sys_msg = """
# You are an expert in vocational training and skills taxonomy development. Your task is to extract skills from a Unit of Competency (UOC) and return them in structured JSON format for integration into a skill database. extract individual **skills**, making sure they are generalizable **capabilities** — not task descriptions.

# Analyze the complete UOC, including:
# •	Application section: for context and domain relevance
# •	Elements and Performance Criteria: for observable and demonstrable skills
# •	Performance Evidence: for real-world task-oriented skills
# •	Knowledge Evidence: for cognitive and knowledge-based components
# •	Foundation Skills: for transferable and enabling skills

# For each extracted skill, provide:
# •	Skill Name: Use a clear, standardized, **similar to skills in job descriptions**, Noun-based name (Verb or Action-based where appropriate)
# •	Skill Description: Brief explanation of what the skill enables the learner to do
# •	Skill Type: One or more of the following:
# Technical, Interpersonal, Cognitive, Administrative, Cultural, Communication, Teamwork, Procedural, Ethical, Analytical, Digital
# •	Skill Domains (multiple applicable): Functional areas where the skill applies, e.g.,
# Health Services, Communication, Aboriginal Cultural Support, Client Support, Advocacy, Scheduling, Documentation
# •	Skill Industries (multiple applicable): Broader industries where the skill is useful, e.g.,
# Health Care, Community Services, Indigenous Health, Aged Care, Disability Services, Public Health, Social Work
# •	Source: Location within the UOC where this skill is derived
# (e.g., Performance Evidence, Knowledge Evidence, Foundation Skills)

# IMPORTANT: Do NOT extract tasks, activities, or instructions. Focus on the **underlying skill or ability** demonstrated by the task.

# Output Format (Example):
# Return your answer in strict JSON format as shown below:
# ```
# [
#   {
#     "skill_name": "Culturally appropriate communication",
#     "description": "The ability to communicate in a manner that is respectful of Aboriginal and/or Torres Strait Islander cultural values and norms.",
#     "skill_type": ["Interpersonal", "Cultural"],
#     "skill_domains": ["Aboriginal Cultural Support", "Communication", "Client Support"],
#     "skill_industries": ["Indigenous Health", "Community Services", "Public Health"],
#     "source": "Application"
#   },
#   {
#     "skill_name": "Appointment Coordination",
#     "description": "The ability to organize, schedule, and confirm appointments and logistical support for client access to health services.",
#     "skill_type": ["Administrative", "Communication", "Digital"],
#     "skill_domains": ["Health Services", "Scheduling", "Client Support"],
#     "skill_industries": ["Health Care", "Community Services", "Aged Care", "Disability Services"],
#     "source": "Performance Evidence / Foundation Skills"
#   }
# ]
# ```

# Guidelines:
# o	Extract 40–45 unique noun-based skills
# o	Make sure all the keys in the JSON are generated. Keys are: skill_name, description, skill_type, skill_domains, skill_industries, source
# o	If a performance criterion states a **task** like “organise transport,” extract the skill as **“Logistics Coordination”**, not the verbatim phrase.
# o	Group multiple task variations under the same skill if conceptually identical (e.g., “follow-up” and “post-service contact” → “Client Follow-up Skills”).
# o	Avoid skills that are just keywords (e.g., “communication”) without context. Add modifiers: “Culturally appropriate communication”.
# o	Apply multiple domains and industries per skill if relevant
# o	Keep the JSON strictly valid — suitable for direct parsing
# o	Focus on observable, transferable, cognitive, and administrative skills across all evidence

# Note: only output the JSON, do not generate any extra sentences.

# """
# o	Avoid verbs in skill names — use skills-as-categories
# sys_msg = """
# You are an expert curriculum analyst. Your task is to extract skills from a Unit of Competency (UOC) for the purpose of building a structured skills taxonomy.
# Analyze the full UOC text, including:
# •	Application: Identify contextual or domain-specific capabilities.
# •	Elements and Performance Criteria: Extract actionable and observable skills.
# •	Performance Evidence: Identify practical, task-based skills.
# •	Knowledge Evidence: Identify cognitive or theoretical knowledge areas that support skill performance.
# •	Foundation Skills: Include literacy, numeracy, tech, and team skills even if not explicitly stated elsewhere.
# Your output should be a structured list of skill statements with the following fields:
# •	Skill Name: A concise and standardized label for the skill
# •	Skill Description: A brief summary of what the learner must be able to do
# •	Skill Type: One or more of the following: Technical, Interpersonal, Cognitive, Administrative, Cultural, Ethical, Communication, Teamwork, Procedural
# •	Skill Domain: Broad category of expertise or application area (e.g., Health Services, Communication, Aboriginal Cultural Support, Administration, Client Services)
# •	Source: The section where the skill was derived (e.g., Knowledge Evidence, Performance Evidence)
# Example Output:
# yaml

# - Skill Name: Use culturally appropriate communication  
#   Description: Communicate with clients using language and behavior that aligns with Aboriginal cultural values and norms.  
#   Skill Type: Interpersonal, Cultural  
#   Skill Domain: Aboriginal Cultural Support  
#   Source: Application

# - Skill Name: Schedule client appointments  
#   Description: Coordinate health service bookings including time, location, and client needs.  
#   Skill Type: Administrative  
#   Skill Domain: Health Services Administration  
#   Source: Performance Evidence / Foundation Skills

# Guidelines:
# •	Identify at least 15 unique skills.
# •	Group similar sub-skills under the same "skill name" if they share intent.
# •	Avoid rephrasing duplicates.
# •	Use consistent naming and avoid jargon in labels.
# •	De-duplicate overlapping or similar skills.


# """

# COMMAND ----------

prompt_v = '4'
sys_msg = """
You are an expert human resources analyst specializing in skill taxonomy development. Your task is to extract human-ability implied skills from job descriptions and convert them into a standardized, taxonomy-friendly format.

Instructions:
Analyze the provided job description and extract skills that represent actual human abilities and competencies. Strictly follow these guidelines:
What to Extract:
    - Human abilities and competencies that can be developed and applied across contexts
    - Contextual skills that specify how generic abilities are applied (e.g., "written communication for technical documentation" rather than just "communication")
    - Underlying capabilities implied by tools or technologies mentioned
What to Avoid:
    - Generic keywords without context (avoid: "communication", "teamwork", "leadership")
    - Tool names without human ability context (avoid: "Excel", "Python", "Photoshop")
    - Company-specific processes or proprietary systems
    - Personal attributes that cannot be learned or developed
Transformation Rules:
1.	Tool/Technology → Human Ability: 
    - "Excel" → "spreadsheet data analysis and modeling"
    - "Python" → "programming for data analysis and automation"
    - "Photoshop" → "digital image editing and design"
2.	Generic Skills → Contextual Skills: 
    - "Communication" → "written communication for stakeholder reporting"
    - "Leadership" → "cross-functional team leadership and coordination"
    - "Problem-solving" → "analytical problem-solving for business optimization"
3.	Standardization Guidelines: 
    - Use present tense, active voice
    - Follow gramatical pattern: [action or ability] + [context or domain] + [purpose if relevant]
    - ALIGN with established skill frameworks (O*NET, ESCO, LinkedIn Skills)
    - Use natural language that humans would recognize

Chain of Thought Process:
Before providing the final JSON output, work through this analysis:
1.	Initial Scan: Identify all potential skill-related terms, tools, and responsibilities
2.	Categorization: Group findings into: 
    - Technical abilities
    - Cognitive abilities
    - Interpersonal abilities
    - Domain-specific knowledge applications
3.	Transformation: Convert each item using the transformation rules above
4.	Validation: Ensure each extracted skill is: 
    - Learnable/developable
    - Transferable to other contexts
    - Specific enough to be meaningful
    - Grammatically consistent with skill taxonomy standards
5.	Deduplication: Remove redundant or overlapping skills

Output Format:
Provide your analysis and final results in this strict JSON format as shown below:
For each skill, output the following fields in JSON format:
[
    {
        "skill": "standardized skill name",
        "shortened_skill": "condensed 2-4 words version for cross-job comparison and common skill identification",
        "category": "technical|cognitive|interpersonal|domain_knowledge",
        "source_context": "original phrase or requirement from job description",
        "confidence": "high|medium|low"
    }
]

Additional guidance:
- Keep the JSON strictly valid — suitable for direct parsing.

Example:
Job Description: "Manage client relationships using Salesforce CRM, create reports in Excel, and present findings to executive team."
Expected Output:
[
    {
        "skill": "customer relationship management using CRM systems",
        "shortened_skill": "CRM administration",
        "category": "technical",
        "source_context": "Manage client relationships using Salesforce CRM",
        "confidence": "high"
    },
    {
        "skill": "spreadsheet data analysis and reporting",
        "shortened_skill": "spreadsheet analysis",
        "category": "technical", 
        "source_context": "create reports in Excel",
        "confidence": "high"
    },
    {
        "skill": "executive-level presentation and communication",
        "shortened_skill": "executive presentation",
        "category": "interpersonal",
        "source_context": "present findings to executive team",
        "confidence": "high"
    }
]

Note: only output the JSON, do not generate any extra sentences.
"""

# COMMAND ----------

# prompt_v = '1'
# sys_msg = """
# You are an expert in text semantic analysis. Given a skill as input:

# TASK 1: Tag the input skill
# Output "YES" if the input skill is aligned with ALL the rules for the skills definition mentioned below, outherwise output "NO". 
# The rules for the skills definition:
# 1. The skill represents a human ability, not a tool, software, or technology.
# 2. It can be acquired or improved through learning and practice.
# 3. It involves applying knowledge, experience, and personal attributes.
# 4. It is dynamic and influenced by context, interaction with others, and environmental demands.
# 5. It is purpose-driven and considered valuable in work or life.

# TASK 2: Output format
# For the input skill, Return your answer in strict JSON format as shown below:
# [
#   {
#     "is_aligned": "<YES or NO>",
#     "reason": "<Output the items that the given skill is NOT aligned with, e.g 1, 3. Otherwise, If all items aligned output 'NA'>"
#   }
# ]

# Additional guidance:
# - Keep the JSON strictly valid — suitable for direct parsing

# Note: only output the JSON, do not generate any extra sentences.

# """

# COMMAND ----------

prompt_v = "cl_6"
sys_msg = """
You are a skill extraction expert specializing in identifying human capabilities implied in unit of competencies (UOC) descriptions. Your task is to analyze the given unit of competency description and extract skills that represent human abilities rather than just keywords or tools.

## Chain of Thought Process:

1. **Read and Analyze**: Carefully read the entire UOC
2. **Identify Context**: Look for tasks, responsibilities, and requirements that imply human abilities
3. **Tool-to-Skill Translation**: Convert tool/technology mentions into the underlying human skills required
4. **Contextualize Generic Terms**: Add specific context to broad terms like "communication"
5. **Standardize Language**: Ensure skills align with established VET and Higher Education taxonomies
6. **Create Variations**: Generate both detailed and shortened versions for taxonomy building

## Extraction Guidelines:

### Do:
- Extract skills that represent **human abilities** and **cognitive processes**
- Convert tools/technologies to underlying skills (e.g., "Excel" → "spreadsheet data analysis and modeling")
- Add specific context to generic skills (e.g., "communication" → "technical documentation writing" or "stakeholder presentation delivery")
- Use standardized terminology consistent with educational frameworks (Bloom's taxonomy, Australian Qualifications Framework, European Qualifications Framework)
- Focus on transferable skills that can be applied across roles
- Include both technical and soft skills with proper context

### Don't:
- List bare tool names without skill context
- Include vague, uncontextualized skills like "communication," "teamwork," or "problem-solving"
- Extract company-specific processes that aren't transferable
- Include personality traits or attitudes
- Use inconsistent grammatical structures

### Skill Naming Convention:
- Use noun phrases that describe the ability
- Follow pattern: <[Action or Process] + [Domain or Context] + [Object or Outcome]>
- Examples: "financial data analysis," "cross-functional team coordination," "regulatory compliance assessment"

## Output Format:
Provide your analysis and final results in this strict JSON format as shown below:
For each skill, output the following fields in JSON format:
[
    {
    "skill_name": "specific detailed skill name",
    "category": "technical|cognitive|interpersonal|domain_knowledge",
    "description": "brief description of the skill application in context",
    "source_context": "the specific part of UOC this was derived from",
    "confidence": "high|medium|low"
    }
]

## Additional guidance:
- Keep the JSON strictly valid — suitable for direct parsing.

## Example Input/Output:

**Input**: "Marketing Manager role requiring proficiency in Google Analytics, Excel reporting, team leadership, and client presentations..."

**Expected Output Structure**:
- Detailed: "web analytics interpretation," "spreadsheet-based reporting creation," "cross-functional team management," "client-facing presentation delivery"
- Shortened: "data analysis," "reporting," "team leadership," "presentation skills"

Now, please analyze the following UOC and extract skills following these guidelines:
 
"""

# COMMAND ----------

len(all_data)

# COMMAND ----------

list_ = [item['text'] for item in all_data]
model_cat = "GPT"
full_prompts = [instruction_format(sys_msg, "Now analyze the following UOC: " + t, template=model_cat) for t in list_]

sampling_params = SamplingParams(
                                max_tokens=8048,
                                temperature=0.0,
                                top_p=1, 
                                frequency_penalty=0,
                                presence_penalty=0, 
                                n=1, 
                                best_of=1
                                )
outputs = llm.generate(full_prompts, sampling_params=sampling_params)

# COMMAND ----------

# responses = [output.text.strip() for out in outputs for output in out.outputs]
responses = []
pattern = r"\[([^]]+)\]"
i = 0
for out in outputs:
    for output in out.outputs:
        try:
            responses.append(["[" + re.search(pattern, output.text.strip()).group(1) + "]"])
        except:
            i += 1
            responses.append(["""
                              [
                                {
                                "skill_name": "None",
                                "category": "None",
                                "description": "None",
                                "source_context": "None",
                                "confidence": "None"
                                }
                            ]
                              """])
# responses = [["[" + re.search(pattern, output.text.strip()).group(1) + "]"] for out in outputs for output in out.outputs]
# responses
i

# COMMAND ----------

output.text

# COMMAND ----------

len(responses), len(all_data)

# COMMAND ----------

import json
from tqdm import tqdm
null = """
[{
    "skill_name": "None",
    "category": "None",
    "description": "None",
    "source_context": "None",
    "confidence": "None"
}]
"""
all_df = []
k = 0
for i, res in tqdm(enumerate(responses)):
    json_text = res[0] #res.replace('```', '').replace('json', '')
    try:
        df = pd.DataFrame(json.loads(json_text))
    except:
        k += 1
        df = pd.DataFrame(json.loads(null))
    dict_ = all_data[i]
    extr_cols = df.columns.tolist()
    for col in df_mf.columns:
        df[col] = dict_[col]
    # df['course_name'] = dict_['Course Name']
    # df['unit_code'] = dict_['Unit_Code']
    # df['unit_title'] = dict_['Unit_Title']
    df['prompt_version'] = prompt_v
    all_df.append(df)
df_res = pd.concat(all_df)
df_res.reset_index(drop=True, inplace=True)
df_res = df_res[df_mf.columns.tolist() + extr_cols]
# df_res['skill_code'] = df_res.index + 1
# Create the 5-digit string code
# df_res['skill_code'] = df_res['skill_code'].apply(lambda x: f'{x:05d}')
display(df_res.head(100)), k, df_res.shape

# COMMAND ----------

json_text

# COMMAND ----------

df_dict = {'Output': df_res, 'Prompt': pd.DataFrame([[prompt_v, sys_msg]], columns=['Prompt_version', 'Prompt'])}
write_output_to_excel(df_dict, output_folder, f"Qual_Skills_Extr_{model_cat}", prompt_v)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clustering 1

# COMMAND ----------

# MAGIC %run ./EnhancedClustering-v2

# COMMAND ----------

active_models = ["jinaai--jina-embeddings-v4", "Qwen3-Embedding-8B"]
embedders = {active_model: SentenceTransformer(get_snapshot_loc(active_model), trust_remote_code=True) for active_model in active_models}

# COMMAND ----------

clusterer = GridSearchJobSkillsClusterer(
        memory_limit_gb=180,
        batch_size=5000,
        embedding_models = active_models,
        embedders=embedders,
        clustering_algorithms=['kmeans', 'gmm', 'agglomerative']
    )

# COMMAND ----------

algorithms_to_use = ['kmeans', 'gmm']

# COMMAND ----------

df_lc = pd.read_excel(output_folder/ "Qual_Skills_Extr_GPT_cl_6.xlsx", sheet_name='Output')
df_lc.shape, df_lc.columns.tolist()

# COMMAND ----------

df_g = df_lc.groupby(['OSCA'], as_index=False).count()
osca_list = df_g['OSCA'].values.tolist()
osca_list

# COMMAND ----------

output_folder_cluster = Path(output_folder / '0_Clustering_v1')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
all_df = []
for osca in osca_list:
    print(f"-----------------------------Run for {osca} --------------------------------")
    clusterer.best_model = None
    clusterer.best_score = -1
    clusterer.grid_search_results = []
    df_ = df_lc[df_lc['OSCA'] == osca].copy()
    df_.reset_index(drop=True, inplace=True)
    clusterer, labels = enhanced_job_skills_clustering_demo(
        skills=df_['skill_name'].values.tolist(),
        clusterer=clusterer  # Will test multiple models
    )
    
    # cl_results = job_skills_ensemble_clustering(df_['skill_name'].values.tolist(), clusterer, algorithms=algorithms_to_use, remaining_algo='kmeans', cluster_on_sim=False)# algo='gmm')
    #job_skills_clustering(df_['skill_name'].values.tolist(), clusterer)
    df_['cluster_id'] = labels# cl_results.best_model['labels']
    df_dict = {'Clustering_Output': df_}
    write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Extr_Enhv4_Clustered_{"GPT"}_{osca.replace(' ', '_')}", 'cl_6')
    break

# COMMAND ----------

display(df_.groupby('cluster_id', as_index=False).count().sort_values(by=['OSCA']))

# COMMAND ----------

display(df_[df_['cluster_id'] == 44][['skill_name']])

# COMMAND ----------

# MAGIC %md
# MAGIC # Deduplications in Clusters

# COMMAND ----------

# This has been used
prompt_v = 'cl_2'
def get_sys_msg(osca):
  return """
You are tasked with analyzing a list of skills to identify and extract distinct, non-duplicated skills. Your goal is to group similar skills together and return the most representative defined skill from each significant group.

Apply these strict instructions:
Step 1: Analyze and Group Skills
Carefully read through the provided list of skills and identify similarities. Group skills that represent the same or closely related competencies, even if they use different wording. 
Consider:
- Synonymous terms (e.g., "leadership" and "team leadership")
- Different levels of specificity (e.g., "Python programming" and "Python")
- Variations in phrasing (e.g., "project management" and "managing projects")
- Skills that fall under the same broader category or domain

Step 2: Apply Chain of Thought Reasoning
For each skill, think through:
- What core competency or domain does this skill represent?
- Which other skills in the list share this same core competency?
- What is the most generalized and concise way to describe this competency?

Step 3: Generate New Representative Skill
For each group:
- Output the skill name that best represents the entire group
- Avoid listing multiple skills together (e.g., no "and" or commas)
- Ensure the output skills clearly communicates the overarching competency
- Ensure to output at least one skill
Context Enhancement:
•	"Communication" → "Professional written communication" or "Cross-functional team communication"
•	"Leadership" → "Team leadership and supervision" or "Strategic organizational leadership"
•	"Problem solving" → "Analytical problem-solving methodology" or "Creative problem resolution"
Grammatical Structure Standards:
•	Use noun phrases with descriptive modifiers
•	Ensure consistency with ESCO (European Skills, Competences, Qualifications and Occupations), O*NET, or similar frameworks
•	Use present participle forms for process skills (e.g., "analyzing", "developing", "managing")
•	Use concrete nouns for knowledge domains (e.g., "financial analysis", "risk assessment")

Note: If no clear representative skill can be generated, default to a general domain-based skill (e.g., "communication", "safety compliance", "risk assessment")

Step 4: Format Output
Return only a JSON array containing the condensed distinct skills, with no additional text or explanation. Use the format below for direct parsing:
[{"skill": "skill_name_here"}]

Your Task:
Process the following list of skills:

"""

# COMMAND ----------

prompt_v = 'cl_2_2'
def get_sys_msg(osca):
  return """
You are tasked with analyzing a list of skills to identify and extract distinct, non-duplicated skills. Your goal is to group similar skills together and return the most representative and broadly defined skill from each significant group.

Instructions:
Step 1: Analyze and Group Skills
Carefully read through the provided list of skills and identify similarities. Group skills that represent the same or closely related competencies, even if they use different wording. Consider:
Synonymous terms (e.g., "leadership" and "team leadership")
Different levels of specificity (e.g., "Python programming" and "Python")
Variations in phrasing (e.g., "project management" and "managing projects")
Skills that fall under the same broader category or domain

Step 2: Apply Chain of Thought Reasoning
For each skill, think through:
What core competency or domain does this skill represent?
Which other skills in the list share this same core competency?
What is the most generalized and concise way to describe this competency?

Step 3: Generate New Representative Skills
For each group:
Generate skill name that best represents the entire group
Avoid listing multiple skills together (e.g., no "and" or commas)
Ensure the output skills clearly communicates the overarching competency
Ensure to output at least one skill

Step 4: Format Output
Return only a JSON array containing the condensed distinct skills, with no additional text or explanation. Use the format below for direct parsing:
[{"distinct_skill": "skill_name_here"}]

Example:
Input Skills:
- identifying ethical industry practices and conducting day‑to‑day activities accordingly
- workplace safety and hygiene compliance
- end‑of‑shift procedural compliance
- personal presentation and hygiene compliance
- follow organisational food hygiene procedures
- identify and assess food safety hazards
- report hygiene non‑compliance and unsafe practices
- implement personal protective equipment and uniform hygiene
- prevent cross‑contamination from clothing and personal items
- avoid direct contact with ready‑to‑eat foods
- apply cleaning and sanitising techniques to food contact surfaces
- perform hand‑washing at critical control points
- communicate personal health issues that pose a hygiene risk
- cease food handling activities when personal health risk is present
- deliver oral reports of hygiene hazards to supervisors
- personal safety risk assessment and assistance coordination
- health and safety information communication to personnel
- safety procedure compliance monitoring
- hazard identification scheduling and execution
- risk control implementation and effectiveness evaluation
- accurate WHS record creation, storage and legal compliance
- high‑level WHS report writing for regulatory compliance
- application of organisational health, safety and security procedures
- hazard identification and removal in the workplace
- personal protective equipment selection and usage
- unsafe work practice reporting and documentation
- communication of health, safety and security concerns to supervisors and WHS representatives

Expected Output:
[
  {"distinct_skill": "ethical work practices"},
  {"distinct_skill": "food safety compliance"},
  {"distinct_skill": "personal hygiene standards"},
  {"distinct_skill": "workplace safety procedures"},
  {"distinct_skill": "hazard identification"},
  {"distinct_skill": "risk management"},
  {"distinct_skill": "regulatory reporting"},
  {"distinct_skill": "health and safety communication"},
  {"distinct_skill": "personal protective equipment usage"}
]

Your Task:
Process the following list of skills:

"""

# COMMAND ----------

prompt_v = 'cl_2_3'
def get_sys_msg(osca):
  return """
You are tasked with analyzing a list of skills to identify and extract distinct, non-duplicated skills. Your goal is to group similar skills together and return the most representative and broadly defined skill from each significant group.

Instructions:
Step 1: Analyze and Group Skills
Carefully read through the provided list of skills and identify similarities. Group skills that represent the same or closely related competencies, even if they use different wording. 
Consider:
Synonymous terms (e.g., "leadership" and "team leadership")
Different levels of specificity (e.g., "Python programming" and "Python")
Variations in phrasing (e.g., "project management" and "managing projects")
Skills that fall under the same broader category or domain

Step 2: Apply Chain of Thought Reasoning
For each skill, think through:
What core competency or domain does this skill represent?
Which other skills in the list share this same core competency?
What is the most generalized and concise way to describe this competency?

Step 3: Generate New Representative Skills
For each group:
Generate skill name that best represents the entire group
Avoid listing multiple skills together (e.g., no "and" or commas)
Ensure the output skills clearly communicates the overarching competency
Ensure to output at least one skill

Context Enhancement:
•	"Communication" → "Professional written communication" or "Cross-functional team communication"
•	"Leadership" → "Team leadership and supervision" or "Strategic organizational leadership"
•	"Problem solving" → "Analytical problem-solving methodology" or "Creative problem resolution"

Grammatical Structure Standards:
•	Use noun phrases with descriptive modifiers
•	Ensure consistency with ESCO (European Skills, Competences, Qualifications and Occupations), O*NET, or similar frameworks
•	Use present participle forms for process skills (e.g., "analyzing", "developing", "managing")
•	Use concrete nouns for knowledge domains (e.g., "financial analysis", "risk assessment")

Step 4: Format Output
Return only a JSON array containing the condensed distinct skills, with no additional text or explanation. Use the format below for direct parsing:
[{"distinct_skill": "skill_name_here"}]

Example:
Input Skills:
- identifying ethical industry practices and conducting day‑to‑day activities accordingly
- workplace safety and hygiene compliance
- end‑of‑shift procedural compliance
- personal presentation and hygiene compliance
- follow organisational food hygiene procedures
- identify and assess food safety hazards
- report hygiene non‑compliance and unsafe practices
- implement personal protective equipment and uniform hygiene
- prevent cross‑contamination from clothing and personal items
- avoid direct contact with ready‑to‑eat foods
- apply cleaning and sanitising techniques to food contact surfaces
- perform hand‑washing at critical control points
- communicate personal health issues that pose a hygiene risk
- cease food handling activities when personal health risk is present
- deliver oral reports of hygiene hazards to supervisors
- personal safety risk assessment and assistance coordination
- health and safety information communication to personnel
- safety procedure compliance monitoring
- hazard identification scheduling and execution
- risk control implementation and effectiveness evaluation
- accurate WHS record creation, storage and legal compliance
- high‑level WHS report writing for regulatory compliance
- application of organisational health, safety and security procedures
- hazard identification and removal in the workplace
- personal protective equipment selection and usage
- unsafe work practice reporting and documentation
- communication of health, safety and security concerns to supervisors and WHS representatives

Expected Output:
[
  {"distinct_skill": "demonstrating ethical conduct in operational activities"},
  {"distinct_skill": "complying with food safety and hygiene standards"},
  {"distinct_skill": "maintaining personal hygiene in food handling environments"},
  {"distinct_skill": "adhering to workplace health and safety procedures"},
  {"distinct_skill": "conducting workplace hazard identification"},
  {"distinct_skill": "implementing operational risk assessment and control"},
  {"distinct_skill": "preparing regulatory health and safety documentation"},
  {"distinct_skill": "communicating health and safety information to stakeholders"},
  {"distinct_skill": "selecting and using personal protective equipment"},
  {"distinct_skill": "complying with shift closure procedures"}
]


Your Task:
Process the following list of skills:
"""

# COMMAND ----------

prompt_v = 'cl_2_4'
def get_sys_msg(osca):
  return """
You are tasked with analyzing a list of skills to identify and extract distinct, non-duplicated skills. Your goal is to group similar skills together and return the most representative skill from each significant group.

Instructions:
Step 1: Analyze and Group Skills
First, carefully read through the provided list of skills and identify similarities. Group skills that represent the same or very similar competencies, even if they use different wording. Consider:
•	Synonymous terms (e.g., "leadership" and "team leadership")
•	Different levels of specificity (e.g., "Python programming" and "Python")
•	Variations in phrasing (e.g., "project management" and "managing projects")
•	Skills that fall under the same broader category

Step 2: Apply Chain of Thought Reasoning
For each skill, think through:
•	What core competency does this skill represent?
•	Which other skills in the list share this same core competency?
•	What would be the most comprehensive yet concise way to describe this competency?

Step 3: Identify and Exclude Outliers
•	Count the number of skills in each group
•	Identify groups with the smallest number of members (outliers)
•	Exclude these outlier groups from your final output, as they represent less significant or potentially irrelevant skills

Step 4: Select Representative Skills
For each remaining group (after excluding outliers):
•	Choose the skill name that best represents the entire group
•	Prefer more comprehensive terms over overly specific ones
•	Ensure the chosen skill clearly communicates the competency
•	Ensure to output at least one skill
•	Apply proper skill taxonomy grammar (see Grammar Guidelines below)

Step 5: Grammar and Format Validation
Ensure all final skills follow established taxonomy naming conventions:
Grammar Guidelines:
Follow these established skill taxonomy grammar rules when naming skills:
  1. Use Gerund Forms (-ing) for Action Skills:
    •	"Reading comprehension", "Critical thinking", "Problem solving"
    •	Format: [Action-ing] + [Object/Domain]
  2. Noun Phrases for Knowledge Areas:
    •	"Mathematics knowledge", "Customer service skills", "Safety procedures"
    •	Format: [Domain] + [knowledge/skills/procedures]
  3. Descriptive Action Phrases:
    •	"Understanding written sentences and paragraphs"
    •	Format: [Action] + [Specific Object] + [Context if needed]
  4. Professional Language Requirements:
    •	Use industry-standard terminology
    •	Avoid colloquialisms or informal language
  •	Maintain consistency in terminology
  5. Action-Oriented Phrasing:
    •	Focus on what can be performed or demonstrated
    •	Use active rather than passive voice
    •	Emphasize competencies that can be assessed
  6. Recommended Pattern:
    [Action/Gerund] + [Object/Domain] + [Context if needed]

Output Format:
Return only a JSON array containing the skills, with no additional text or explanation. Use the format below for direct parsing:
[
  {"skill": "skill_name_1"},
  {"skill": "skill_name_2"},
  {"skill": "skill_name_3"}
]

Final Validation Checklist:
Before finalizing your output, verify that each skill:
•	Follows proper grammar conventions from the guidelines
•	Uses professional, industry-standard terminology
•	Is action-oriented and measurable
•	Represents a comprehensive competency area
•	Is completely distinct from other selected skills
•	Covers all relevant activities from the original list

Your Task:
Process the following list of skills:

"""

# COMMAND ----------

# This has been used
prompt_v = 'cl_2_1'
def get_sys_msg(osca):
  return """
You are given a list of skills. Your task is to (1) group skills that represent the same competency, (2) exclude outlier groups, and (3) return a JSON array of the most representative (canonical) skill for each remaining group.

STRICT OUTPUT:
- Return ONLY a valid JSON array (no extra text, no comments).
- Each element must be an object of the form: {"distinct_skill": "skill_name"}.
- Use double quotes, no trailing commas, valid JSON syntax.

INPUT:
- skills: an array of strings (see below).

PARAMETERS:
- min_group_size = 2 (exclude groups smaller than this).
- Always output at least one skill. If all groups are singletons, output exactly one canonical skill (see Tie-breakers).

DEFINITIONS:
- “Group”: a set of skills that denote the same core competency.
- “Canonical skill”: the clearest, most comprehensive, and widely recognized label for a group.

PROCESS (reason internally; do not reveal your reasoning):
1) PREPROCESS & NORMALIZE
   - Trim whitespace; collapse multiple spaces.
   - Case-normalize for matching (but preserve proper casing in final output).
   - Remove trivial suffixes like “skills”, “experience”, “proficiency” if they don’t change meaning.
   - Standardize separators (e.g., “PowerBI” → “Power BI”; “C#/.NET” split/interpret as distinct if needed).
   - Singularize nouns where appropriate (“Dashboards” → “Dashboard”), unless brand names or established plurals.
   - Treat common acronyms and aliases as candidates for the same group (e.g., “JS” ↔ “JavaScript”, “NLP” ↔ “Natural Language Processing”), but do not invent expansions not supported by context.

2) GROUP SIMILAR SKILLS (semantic + lexical)
   - Merge items that are synonyms, aliases, abbreviations, or minor phrasing variants:
     • Synonyms: “leadership”, “team leadership”
     • Phrasing variants: “project management”, “managing projects”
     • Specific vs general: “Python programming” ↔ “Python”
     • Spelling/casing variants: “Power BI”, “PowerBI”
     • Acronym ↔ full: “JS” ↔ “JavaScript”
   - Do not merge distinct technologies or materially different competencies (e.g., “Python” ≠ “R”; “Data analysis” ≠ “Data visualization” unless clearly the same).

3) EXCLUDE OUTLIER GROUPS
   - Count group sizes.
   - Remove groups with size < min_group_size.
   - If this removes all groups, keep exactly one canonical skill chosen via Tie-breakers.

4) SELECT CANONICAL (REPRESENTATIVE) SKILL PER GROUP
   - Prefer the most comprehensive, clear, and widely recognized label.
   - Prefer generic competency label over overly specific variants (e.g., choose “Python” or “Python programming” over “Python scripting for data pipelines”).
   - Prefer standard names for technologies, frameworks, and certifications.
   - Keep brand capitalization and formatting (e.g., “JavaScript”, “Power BI”, “C++”, “Node.js”).
   - Use singular form unless the proper noun is plural (e.g., “Tableau”, not “Tableaus”).

5) TIE-BREAKERS (deterministic)
   - If two labels are equally suitable:
     a) Choose the one that is more widely recognized/standard (e.g., “JavaScript” over “JS”).
     b) If still tied, prefer the more general term (e.g., “Python” over “Python programming”).
     c) If still tied, choose the one that appears earliest in the input list.
   - If all groups are singletons, return exactly one skill using the same tie-breakers across the entire input.

OUTPUT FORMAT:
- Return only:
  [
    {"distinct_skill": "Canonical Skill 1"},
    {"distinct_skill": "Canonical Skill 2"}
  ]

EXAMPLE:
Input Skills:
- identifying ethical industry practices and conducting day‑to‑day activities accordingly
- workplace safety and hygiene compliance
- end‑of‑shift procedural compliance
- personal presentation and hygiene compliance
- follow organisational food hygiene procedures
- identify and assess food safety hazards
- report hygiene non‑compliance and unsafe practices
- implement personal protective equipment and uniform hygiene
- prevent cross‑contamination from clothing and personal items
- avoid direct contact with ready‑to‑eat foods
- apply cleaning and sanitising techniques to food contact surfaces
- perform hand‑washing at critical control points
- communicate personal health issues that pose a hygiene risk
- cease food handling activities when personal health risk is present
- deliver oral reports of hygiene hazards to supervisors
- personal safety risk assessment and assistance coordination
- health and safety information communication to personnel
- safety procedure compliance monitoring
- hazard identification scheduling and execution
- risk control implementation and effectiveness evaluation
- accurate WHS record creation, storage and legal compliance
- high‑level WHS report writing for regulatory compliance
- application of organisational health, safety and security procedures
- hazard identification and removal in the workplace
- personal protective equipment selection and usage
- unsafe work practice reporting and documentation
- communication of health, safety and security concerns to supervisors and WHS representatives

output:
[ 
    {"distinct_skill": "Food safety and hygiene"}, 
    {"distinct_skill": "WHS hazard identification and risk management"}, 
    {"distinct_skill": "WHS recordkeeping and regulatory reporting"}, 
    {"distinct_skill": "Personal protective equipment (PPE) selection and use"}, 
    {"distinct_skill": "WHS communication and incident reporting"}, 
    {"distinct_skill": "Health, safety and security procedure compliance"} 
]

NOW PROCESS THIS INPUT:

"""

# COMMAND ----------

prompt_v = 'cl_old'
def get_sys_msg(osca):
  return """
You are tasked with analyzing a list of skills to identify and extract distinct, non-duplicated skills. Your goal is to group similar skills together and return the most representative skill from each significant group.

Instructions:
Step 1: Analyze and Group Skills First, carefully read through the provided list of skills and identify similarities. Group skills that represent the same or very similar competencies, even if they use different wording. Consider:
•	Synonymous terms (e.g., "leadership" and "team leadership")
•	Different levels of specificity (e.g., "Python programming" and "Python")
•	Variations in phrasing (e.g., "project management" and "managing projects")
•	Skills that fall under the same broader category
Step 2: Apply Chain of Thought Reasoning For each skill, think through:
•	What core competency does this skill represent?
•	Which other skills in the list share this same core competency?
•	What would be the most comprehensive yet concise way to describe this competency?
Step 3: Identify and Exclude Outliers
•	Count the number of skills in each group
•	Identify groups with the smallest number of members (outliers)
•	Exclude these outlier groups from your final output, as they represent less significant or potentially irrelevant skills
•	Make sure to output at least one skill.
Step 4: Select Representative Skills For each remaining group (after excluding outliers):
•	Choose the skill name that best represents the entire group
•	Prefer more comprehensive terms over overly specific ones
•	Ensure the chosen skill clearly communicates the competency
Step 5: Format Output Return only a JSON array containing the distinct skills, with no additional text or explanation. Provide your analysis and final results in this strict JSON format as shown below for direct parsing:
[{"distinct_skill": "skill_name_here"}]

Example Process:
Input Skills: ["Python", "Python programming", "JavaScript", "JS", "Team leadership", "Leadership", "Data analysis"]
Chain of Thought:
•	Group 1 (Python skills): "Python", "Python programming" → Best representative: "Python programming"
•	Group 2 (JavaScript skills): "JavaScript", "JS" → Best representative: "JavaScript"
•	Group 3 (Leadership skills): "Team leadership", "Leadership" → Best representative: "Team leadership"
•	Group 4 (Analysis skills): "Data analysis" → Only one member, this is an outlier group
After excluding outliers: Keep groups 1, 2, and 3 (each has 2 members), exclude group 4 (only 1 member)
Expected Output:
[
  {"distinct_skill": "Python programming"},
  {"distinct_skill": "JavaScript"},
  {"distinct_skill": "Team leadership"}
]
Your Task:
Process the following list of skills and return the distinct skills in the specified JSON format:


"""

# COMMAND ----------

input_folder_cluster = Path(output_folder / '0_Clustering_v1')
output_folder_cluster = Path(output_folder / '1_Cluster_Dedup_v1')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
count = 10
for f in list(input_folder_cluster.glob('*.xlsx')):
    df_ = pd.read_excel(f)#.drop(['posting_text'], axis='columns')
    df_agg = df_[df_['cluster_id'] != -1].groupby(['OSCA', 'cluster_id'], as_index=False)['skill_name'].apply(lambda x: '\n'.join([f"- {s}" for s in x.head(10000)]))
    all_data = df_agg.to_dict(orient='records')
    osca = df_['OSCA'].values.tolist()[0]
    # if osca in ['Accounts Clerk','Barista'] : continue
    print(f"----------{osca}-----------")
    counter = 1
    loop = True
    temp = 0.0
    while loop and (counter <= count):
        loop = False
        try:
            sys_msg = get_sys_msg(osca)
            list_ = [item['skill_name'] for item in all_data]
            model_cat = "GPT"
            full_prompts = [instruction_format(sys_msg, "Input Skills: \n" + t, template=model_cat) for t in list_]

            sampling_params = SamplingParams(
                                            max_tokens=8048,
                                            temperature=temp,
                                            top_p=1, 
                                            frequency_penalty=0,
                                            presence_penalty=0, 
                                            n=1, 
                                            best_of=1
                                            )
            outputs = llm.generate(full_prompts, sampling_params=sampling_params)
            responses = []
            pattern = r"assistantfinal\[([^]]+)\]"
            i = 0
            for out in outputs:
                for output in out.outputs:
                    responses.append(["[" + re.search(pattern, output.text.strip()).group(1) + "]"])
            all_df = []
            for i, res in tqdm(enumerate(responses)):
                json_text = res[0] #res.replace('```', '').replace('json', '')
                df = pd.DataFrame(json.loads(json_text))

                dict_ = all_data[i]
                extr_cols = df.columns.tolist()
                for col in df_agg.columns:
                    df[col] = dict_[col]
                df['prompt_version'] = prompt_v
                all_df.append(df)
            df_res = pd.concat(all_df)
            df_res.reset_index(drop=True, inplace=True)
            df_res = df_res[df_agg.columns.tolist() + extr_cols].drop(["skill_name"], axis='columns')
            print(df_res.shape)
            df_dict = {'Cluster_Labeling': df_res}
            write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Extr_Dedup1_{"GPT"}_{osca.replace(' ', '_')}", prompt_v)
        except Exception as e:
            print(f"try {count - counter} more time...{e}, {traceback.format_exc(), output.text.strip()}")
            loop = True
            if counter <= 5:
                temp += 0.1
            counter += 1
    break

# COMMAND ----------

display(df_res)

# COMMAND ----------

import traceback
input_folder_cluster = Path(output_folder / '1_Cluster_Dedup_v1')
output_folder_cluster = Path(output_folder / '2_Cluster_Dedup_v2')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
count = 4
for f in list(input_folder_cluster.glob('*.xlsx')):
    if "Qual_Skills_Extr_Dedup1_GPT_Accounts_Clerk_cl_old" not in str(f): continue
    print(f"Loading file: {f}")
    df_ = pd.read_excel(f)#.drop(['posting_text'], axis='columns')
    # df_agg = df_[df_['cluster_id'] != -1].groupby(['osca_title', 'cluster_id'], as_index=False)['distinct_skill'].apply(lambda x: '\n'.join([f"- {s}" for s in x.head(100)]))
    all_data = df_.to_dict(orient='records')
    osca = df_['OSCA'].values.tolist()[0]
    # if osca not in ['Barista'] : continue
    print(f"----------{osca}-----------")
    counter = 1
    loop = True
    temp = 0.0
    while loop and (counter <= count):
        loop = False
        try:
            sys_msg = get_sys_msg(osca)
            list_ = [item['distinct_skill'] for item in all_data]
            list_1 = ['\n'.join([f"- {s}" for s in list_])]
            model_cat = "GPT"
            full_prompts = [instruction_format(sys_msg, "Skills: \n" + t, template=model_cat) for t in list_1]

            sampling_params = SamplingParams(
                                            max_tokens=8048,
                                            temperature=temp,
                                            top_p=1, 
                                            frequency_penalty=0,
                                            presence_penalty=0, 
                                            n=1, 
                                            best_of=1
                                            )
            outputs = llm.generate(full_prompts, sampling_params=sampling_params)
            responses = []
            pattern = r"assistantfinal\[([^]]+)\]"
            i = 0
            for out in outputs:
                for output in out.outputs:
                    responses.append(["[" + re.search(pattern, output.text.strip()).group(1) + "]"])
            all_df = []
            for i, res in tqdm(enumerate(responses)):
                json_text = res[0] #res.replace('```', '').replace('json', '')
                df = pd.DataFrame(json.loads(json_text))

                dict_ = all_data[i]
                # extr_cols = df.columns.tolist()
                # for col in df_.columns:
                #     df[col] = dict_[col]
                df['OSCA'] = osca
                df['prompt_version'] = prompt_v
                all_df.append(df)
            df_res = pd.concat(all_df)
            df_res.reset_index(drop=True, inplace=True)
            # df_res = df_res[df_.columns.tolist()]#.drop(["distinct_skill"], axis='columns')
            print(df_res.shape)
            df_dict = {'Cluster_Labeling': df_res}
            write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Extr_Dedup2_{"GPT"}_{osca.replace(' ', '_')}", prompt_v)
        except Exception as e:
            print(f"try {count - counter} more time...{e}, {traceback.format_exc()}")
            loop = True
            if counter <= 5:
                temp += 0.1
            counter += 1

# COMMAND ----------

# MAGIC %md
# MAGIC # Clustering 2

# COMMAND ----------

# MAGIC %run  ./EnhancedClustering-v4

# COMMAND ----------

active_models = ["jinaai--jina-embeddings-v4", "Qwen3-Embedding-8B"]
embedders = {active_model: SentenceTransformer(get_snapshot_loc(active_model), trust_remote_code=True) for active_model in active_models}

# COMMAND ----------

clusterer = GridSearchJobSkillsClusterer(
        memory_limit_gb=180,
        batch_size=5000,
        embedding_models = active_models,
        embedders=embedders,
        clustering_algorithms=['kmeans', 'gmm', 'agglomerative']
    )

# COMMAND ----------

algorithms_to_use = ['kmeans', 'gmm']#, 'agglomerative']
# result_clusterer = job_skills_ensemble_clustering(
#     sample_skills, 
#     clusterer, 
#     algorithms=algorithms_to_use
# )

# COMMAND ----------


output_folder_cluster = Path(output_folder / '2_Clustering_v1')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
input_folder = output_folder / "1_Cluster_Dedup_v1"
for f in list(input_folder.glob('*.xlsx')):
    if "Qual_Skills_Extr_Dedup1_GPT_Accounts_Clerk_cl_old" not in str(f): continue
    df_ = pd.read_excel(f).drop('cluster_id', axis='columns')
    osca = df_['OSCA'].values.tolist()[0]
    # if osca not in ['Barista'] : continue
    print(f"-----------------------------Run for {osca} --------------------------------")
    clusterer.best_model = None
    clusterer.best_score = -1
    clusterer.grid_search_results = []
    clusterer, labels = enhanced_job_skills_clustering_demo(
        skills=df_['distinct_skill'].values.tolist(),
        clusterer=clusterer  # Will test multiple models
    )
    # cl_results = job_skills_ensemble_clustering(df_['distinct_skill'].values.tolist(), clusterer, algorithms=algorithms_to_use, remaining_algo='kmeans', cluster_on_sim=False)# algo='gmm')
    df_['cluster_id'] = labels #cl_results.best_model['labels']
    df_dict = {'Clustering_Output': df_}
    write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Lbl_EnsClustered2_{"GPT"}_{osca.replace(' ', '_')}", 'cl_6')

# COMMAND ----------

display(df_.groupby('cluster_id', as_index=False).count().sort_values(by=['OSCA']))

# COMMAND ----------

display(df_[df_['cluster_id'] == 49])

# COMMAND ----------

# MAGIC %md
# MAGIC # Dedup

# COMMAND ----------

import time
while True:
    time.sleep(10)

# COMMAND ----------

prompt_v = 'cl_1'
def get_sys_msg(osca):
  return f"""
You are an expert skills taxonomy analyst tasked with extracting strictly distinct skills from a list of similar or overlapping skills. Your goal is to create a standardized, taxonomy-ready list that aligns with established VET (Vocational Education and Training) and Higher Education curriculum frameworks.

Instructions:
Step 1: Analysis Phase (Chain of Thought)
First, analyze the input skills list by:
  1.	Grouping Similar Skills: Identify skills that refer to the same core competency but use different terminology
  2.	Tool/Technology Abstraction: When specific tools are mentioned (e.g., "Excel", "Photoshop", "Python"), abstract them to the underlying skill domain
  3.	Context Enhancement: Identify vague or generic skills that lack sufficient context
  4.	Taxonomy Alignment: Consider how each skill fits within established educational and professional frameworks
  5.  Exclude skills which are not aligned with **"{osca}"** role"""+"""

Step 2: Extraction Rules
Apply these strict guidelines:
Tool/Technology Abstraction:
•	"Excel analysis" → "Spreadsheet data analysis"
•	"Photoshop design" → "Digital image editing and manipulation"
•	"Python programming" → "Object-oriented programming"
•	"AutoCAD drafting" → "Computer-aided design and drafting"
Context Enhancement:
•	"Communication" → "Professional written communication" or "Cross-functional team communication"
•	"Leadership" → "Team leadership and supervision" or "Strategic organizational leadership"
•	"Problem solving" → "Analytical problem-solving methodology" or "Creative problem resolution"
Grammatical Structure Standards:
•	Use noun phrases with descriptive modifiers
•	Ensure consistency with ESCO (European Skills, Competences, Qualifications and Occupations), O*NET, or similar frameworks
•	Use present participle forms for process skills (e.g., "analyzing", "developing", "managing")
•	Use concrete nouns for knowledge domains (e.g., "financial analysis", "risk assessment")
Distinctness Criteria:
•	No two extracted skills should overlap in scope or application
•	Break down different skills (e.g., "Processing accounts‑payable and accounts‑receivable transactions" → "Processing accounts‑payable transactions", "Processing accounts‑receivable transactions")
•	Each skill must represent a unique competency cluster
•	Avoid redundant variations of the same core skill

Note: Make sure to return at least one skill in the output if you cant the output 'None': [{"skill": "None"}]

Step 3: Output Format
Provide your analysis as structured thoughts, then output ONLY the final JSON array.
Example Process:
Input Skills: ["Excel", "Google Sheets", "Data analysis", "Communication", "Team work", "Leadership"]
Analysis:
•	"Excel" and "Google Sheets" → Both are spreadsheet tools → Abstract to "Spreadsheet data analysis"
•	"Data analysis" → Already abstract, but could be more specific → Keep as "Quantitative data analysis"
•	"Communication" → Too vague → Need context → "Professional interpersonal communication"
•	"Team work" → Collaborative skill → "Collaborative team participation"
•	"Leadership" → Management skill → "Team leadership and supervision"

Output:
Provide your analysis and final results in this strict JSON format as shown below:
For each skill, output the following fields in JSON format:
[
  {"skill": "Spreadsheet data analysis"},
  {"skill": "Quantitative data analysis"},
  {"skill": "Professional interpersonal communication"},
  {"skill": "Collaborative team participation"},
  {"skill": "Team leadership and supervision"}
]

Your Task:
Process the following skills list and provide your chain of thought analysis followed by the JSON output, Remember: The output must be strictly in JSON format as shown above.

"""

# COMMAND ----------

input_folder_cluster = Path(output_folder / '2_Clustering_v1')
output_folder_cluster = Path(output_folder / '2_Cluster_Dedup_v2')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
count = 10
for f in list(input_folder_cluster.glob('*.xlsx')):
    df_ = pd.read_excel(f)#.drop(['posting_text'], axis='columns')
    df_agg = df_[df_['cluster_id'] != -1].groupby(['OSCA', 'cluster_id'], as_index=False)['distinct_skill'].apply(lambda x: '\n'.join([f"- {s}" for s in x.head(1000)]))
    all_data = df_agg.to_dict(orient='records')
    osca = df_['OSCA'].values.tolist()[0]
    # if osca not in ['Barista'] : continue
    print(f"----------{osca}-----------")
    counter = 1
    loop = True
    temp = 0.0
    while loop and (counter <= count):
        loop = False
        try:
            sys_msg = get_sys_msg(osca)
            list_ = [item['distinct_skill'] for item in all_data]
            model_cat = "GPT"
            full_prompts = [instruction_format(sys_msg, "Skills: \n" + t, template=model_cat) for t in list_]

            sampling_params = SamplingParams(
                                            max_tokens=8048,
                                            temperature=temp,
                                            top_p=1, 
                                            frequency_penalty=0,
                                            presence_penalty=0, 
                                            n=1, 
                                            best_of=1
                                            )
            outputs = llm.generate(full_prompts, sampling_params=sampling_params)
            responses = []
            pattern = r"\[([^]]+)\]"
            i = 0
            for out in outputs:
                for output in out.outputs:
                    responses.append(["[" + re.search(pattern, output.text.strip()).group(1) + "]"])
            all_df = []
            for i, res in tqdm(enumerate(responses)):
                json_text = res[0] #res.replace('```', '').replace('json', '')
                df = pd.DataFrame(json.loads(json_text))

                dict_ = all_data[i]
                extr_cols = df.columns.tolist()
                for col in df_agg.columns:
                    df[col] = dict_[col]
                df['prompt_version'] = prompt_v
                all_df.append(df)
            df_res = pd.concat(all_df)
            df_res.reset_index(drop=True, inplace=True)
            df_res = df_res[df_agg.columns.tolist() + extr_cols].drop(["distinct_skill"], axis='columns')
            print(df_res.shape)
            df_dict = {'Cluster_Labeling': df_res}
            write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Extr_Dedup2_{"GPT"}_{osca.replace(' ', '_')}", '5')
        except:
            print(f"try {count - counter} more time...")
            loop = True
            if counter <= 5:
                temp += 0.1
            counter += 1

# COMMAND ----------

display(df_res)

# COMMAND ----------

import traceback
input_folder_cluster = Path(output_folder / '2_Cluster_Dedup_v2')
output_folder_cluster = Path(output_folder / '3_Cluster_Dedup_v3')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
count = 10
for f in list(input_folder_cluster.glob('*.xlsx')):
    df_ = pd.read_excel(f).dropna()#.drop(['posting_text'], axis='columns')
    # df_agg = df_[df_['cluster_id'] != -1].groupby(['osca_title', 'cluster_id'], as_index=False)['distinct_skill'].apply(lambda x: '\n'.join([f"- {s}" for s in x.head(100)]))
    all_data = df_.to_dict(orient='records')
    osca = df_['OSCA'].values.tolist()[0]
    # if osca not in ['Barista'] : continue
    print(f"----------{osca}-----------")
    counter = 1
    loop = True
    temp = 0.0
    while loop and (counter <= count):
        loop = False
        try:
            sys_msg = get_sys_msg(osca)
            list_ = [item['skill'] for item in all_data]
            list_1 = ['\n'.join([f"- {s}" for s in list_])]
            model_cat = "GPT"
            full_prompts = [instruction_format(sys_msg, "Skills: \n" + t, template=model_cat) for t in list_1]

            sampling_params = SamplingParams(
                                            max_tokens=8048,
                                            temperature=temp,
                                            top_p=1, 
                                            frequency_penalty=0,
                                            presence_penalty=0, 
                                            n=1, 
                                            best_of=1
                                            )
            outputs = llm.generate(full_prompts, sampling_params=sampling_params)
            responses = []
            pattern = r"\[([^]]+)\]"
            i = 0
            for out in outputs:
                for output in out.outputs:
                    responses.append(["[" + re.search(pattern, output.text.strip()).group(1) + "]"])
            all_df = []
            for i, res in tqdm(enumerate(responses)):
                json_text = res[0] #res.replace('```', '').replace('json', '')
                df = pd.DataFrame(json.loads(json_text))

                dict_ = all_data[i]
                # extr_cols = df.columns.tolist()
                # for col in df_.columns:
                #     df[col] = dict_[col]
                df['OSCA'] = osca
                df['prompt_version'] = prompt_v
                all_df.append(df)
            df_res = pd.concat(all_df)
            df_res.reset_index(drop=True, inplace=True)
            # df_res = df_res[df_.columns.tolist()]#.drop(["distinct_skill"], axis='columns')
            print(df_res.shape)
            df_dict = {'Cluster_Labeling': df_res}
            write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Extr_Dedup3_{"GPT"}_{osca.replace(' ', '_')}", '5')
        except Exception as e:
            print(f"try {count - counter} more time...{e}, {traceback.format_exc()}")
            loop = True
            if counter <= 5:
                temp += 0.1
            counter += 1

# COMMAND ----------

display(df_res)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cluster Generation and Labeling

# COMMAND ----------

prompt_v = 'cl_4'
def get_sys_msg(osca):
  return """
You are an expert in workforce taxonomy and skills classification. Your task is to cluster a given list of skills and generate standardized, noun-based cluster labels that align with established VET (Vocational Education and Training) and Higher Education curriculum frameworks.

Instructions:
1.	Analyze the skills list systematically: 
    o	Group similar skills based on their core functions, domains, and required competencies
    o	Consider the underlying knowledge areas and practical applications
    o	Identify tools, technologies, and methodologies mentioned
2.	Apply clustering logic: 
    o	Skills requiring similar knowledge bases should be grouped together
    o	Tools/technologies should be generalized (e.g., "Excel" becomes part of "Spreadsheet Data Analysis")
    o	Consider the level of specificity and functional relationships
    o	Single skills can form their own cluster if they don't naturally fit elsewhere
3.	Generate standardized cluster labels: 
    o	Use noun-based phrases that clearly describe the skill domain
    o	Include contextual modifiers to avoid generic terms (e.g., "Financial Reporting" instead of "Reporting")
    o	Align with standard occupational classification systems and educational taxonomies
    o	Ensure grammatical consistency and professional terminology
    o	Make labels to be transferable across different occupations
4.	Quality checks: 
    o	Verify each skill is assigned to exactly one cluster
    o	Ensure cluster labels are descriptive and taxonomy-appropriate
    o	Confirm JSON format is valid for direct parsing
    o	Review for consistency in naming conventions

Chain of Thought Process:
Before providing your final answer, work through this reasoning:
    1.	Initial Analysis: "I will first examine each skill to understand its core function and domain..."
    2.	Clustering Strategy: "Based on the analysis, I can identify these potential groupings... make sure to group more skills together as much as you can"
    3.	Label Generation: "For each cluster, I will create labels that follow taxonomy standards..."
    4.	Validation: "I will verify that each label is appropriately scoped and grammatically consistent..."

Output Format:
Provide your response in valid JSON format as an array of objects, where each object contains:
•	skill: The original skill text (exactly as provided)
•	cluster_label: The standardized noun-based cluster label (a short version to be transferable between occupations)

Example:
Input Skills:
•	Preparing and issuing invoices
•	Processing high-volume invoices and purchase orders
•	Performing bank and credit-card statement reconciliation
•	Analyzing general ledger reconciliations and month-end adjustments
•	providing customer order taking and personalized service
•	creating memorable guest experiences through hospitality
Output:
[
  {"skill": "Preparing and issuing invoices", "cluster_label": "Invoice Management"},
  {"skill": "Processing high-volume invoices and purchase orders", "cluster_label": "Invoice Management"},
  {"skill": "Performing bank and credit-card statement reconciliation", "cluster_label": "Accounts Reconciliation"},
  {"skill": "Analyzing general ledger reconciliations and month-end adjustments", "cluster_label": "Accounts Reconciliation"},
  {"skill": "providing customer order taking and personalized service", "cluster_label": "Customer Service"},
  {"skill": "creating memorable guest experiences through hospitality", "cluster_label": "Customer service"}
]
Now, please provide your chain of thought analysis followed by the JSON output for the given skills list.

"""

# COMMAND ----------

import traceback
input_folder_cluster = Path(output_folder / '3_Cluster_Dedup_v3')
output_folder_cluster = Path(output_folder / '1_Labeling_v2')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
count = 10
for f in list(input_folder_cluster.glob('*.xlsx')):
    df_ = pd.read_excel(f).dropna()#.drop(['posting_text'], axis='columns')
    # df_agg = df_[df_['cluster_id'] != -1].groupby(['osca_title', 'cluster_id'], as_index=False)['distinct_skill'].apply(lambda x: '\n'.join([f"- {s}" for s in x.head(100)]))
    all_data = df_.to_dict(orient='records')
    osca = df_['OSCA'].values.tolist()[0]
    # if osca not in ['Barista'] : continue
    print(f"----------{osca}-----------")
    counter = 1
    loop = True
    temp = 0.0
    while loop and (counter <= count):
        loop = False
        try:
            sys_msg = get_sys_msg(osca)
            list_ = [item['skill'] for item in all_data]
            list_1 = ['\n'.join([f"- {s}" for s in list_])]
            model_cat = "GPT"
            full_prompts = [instruction_format(sys_msg, "Input Skills: \n" + t, template=model_cat) for t in list_1]

            sampling_params = SamplingParams(
                                            max_tokens=8048,
                                            temperature=temp,
                                            top_p=1, 
                                            frequency_penalty=0,
                                            presence_penalty=0, 
                                            n=1, 
                                            best_of=1
                                            )
            outputs = llm.generate(full_prompts, sampling_params=sampling_params)
            responses = []
            pattern = r"\[([^]]+)\]"
            i = 0
            for out in outputs:
                for output in out.outputs:
                    responses.append(["[" + re.search(pattern, output.text.strip()).group(1) + "]"])
            all_df = []
            for i, res in tqdm(enumerate(responses)):
                json_text = res[0] #res.replace('```', '').replace('json', '')
                df = pd.DataFrame(json.loads(json_text))

                dict_ = all_data[i]
                # extr_cols = df.columns.tolist()
                # for col in df_.columns:
                #     df[col] = dict_[col]
                df['OSCA'] = osca
                df['prompt_version'] = prompt_v
                all_df.append(df)
            df_res = pd.concat(all_df)
            df_res.reset_index(drop=True, inplace=True)
            # df_res = df_res[df_.columns.tolist()]#.drop(["distinct_skill"], axis='columns')
            print(df_res.shape)
            df_dict = {'Cluster_Labeling': df_res}
            write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Extr_Labeling_{"GPT"}_{osca.replace(' ', '_')}", prompt_v)
        except Exception as e:
            print(f"try {count - counter} more time...{e}, {traceback.format_exc()}")
            loop = True
            if counter <= 5:
                temp += 0.1
            counter += 1

# COMMAND ----------

display(df_res)

# COMMAND ----------

# MAGIC %md
# MAGIC # Skill Category and Description Generation

# COMMAND ----------

prompt_v = 'cl_5'
def get_sys_msg(osca):
  return """
You are an expert in workforce development and skills analysis. Your task is to analyze a given list of skills and provide brief contextual descriptions lign with established VET (Vocational Education and Training) and Higher Education curriculum frameworks along with appropriate skill category classifications.

Instructions:
1.	Analyze each skill systematically:
    o	Identify the core competency and its practical application
    o	Consider the context in which this skill would be utilized
    o	Determine the primary skill type based on its fundamental nature
2.	Generate skill descriptions:
    o	Write brief, contextual descriptions (10-15 words typical)
    o	Focus on practical application rather than definition
    o	Use action-oriented language that explains "how" or "what" is accomplished
    o	Avoid redundant phrasing with the skill name
    o	Emphasize the practical impact or outcome of applying the skill
    o	Align with standard occupational classification systems and educational taxonomies
    o	Ensure grammatical consistency and professional terminology
3.	Assign skill categories using these definitions:
    o	technical: Skills involving specific tools, systems, equipment, or standardized procedures
    o	cognitive: Skills requiring mental processes like analysis, problem-solving, decision-making, or strategic thinking
    o	interpersonal: Skills involving direct interaction with people, communication, or relationship management
    o	domain_knowledge: Specialized knowledge or expertise specific to an industry, field, or subject area
4.	Category assignment logic:
    o	If a skill primarily involves operating systems or tools → technical
    o	If a skill requires analytical thinking or decision-making → cognitive
    o	If a skill centers on human interaction or communication → interpersonal
    o	If a skill requires specialized field expertise → domain_knowledge
    o	Choose the most dominant characteristic if multiple categories could apply

Chain of Thought Process:
Before providing your final answer, work through this reasoning for each skill:
    1.	Skill Analysis: "This skill involves [core activity/competency]..."
    2.	Context Identification: "It would be applied in situations where [practical context]..."
    3.	Category Logic: "The primary nature of this skill is [reasoning for category choice]..."
    4.	Description Crafting: "The description should emphasize [key practical application]..."

Output Format:
Provide your response in valid JSON format as an array of objects, where each object contains:
•	skill: The original skill text (exactly as provided)
•	category: One of "technical", "cognitive", "interpersonal", or "domain_knowledge"
•	description: Brief contextual description focusing on practical application
Example:
Input Skills:
•	Retail transaction processing and point-of-sale operation
•	Optical dispensing knowledge and lens fitting
•	Confident sales persuasion and closing techniques
•	Customer service problem resolution
Chain of Thought Sample: "Retail transaction processing involves operating POS systems and handling payment procedures - this is primarily technical. The description should focus on the practical tasks involved..."
Output:
[
  {"skill": "Retail transaction processing and point-of-sale operation", "category": "technical", "description": "handling sales payments, issuing receipts and updating inventory records accurately"},
  {"skill": "Optical dispensing knowledge and lens fitting", "category": "domain_knowledge", "description": "understanding prescription verification, lens material selection and proper frame adjustment techniques"},
  {"skill": "Confident sales persuasion and closing techniques", "category": "cognitive", "description": "guiding customers through decision points and securing purchase commitments without pressure"},
  {"skill": "Customer service problem resolution", "category": "interpersonal", "description": "addressing complaints or product issues promptly to maintain satisfaction and store reputation"}
]
Quality Checks:
•	Ensure descriptions are concise and action-focused
•	Verify category assignments align with the primary skill nature
•	Confirm JSON format is valid for direct parsing
•	Review for consistency in description style and length
Now, please provide your chain of thought analysis followed by the JSON output for the given skills list.

"""

# COMMAND ----------

import traceback
input_folder_cluster = Path(output_folder / '1_Labeling_v2')
output_folder_cluster = Path(output_folder / '1_Category_v2')
output_folder_cluster.mkdir(parents=True, exist_ok=True)
count = 10
for f in list(input_folder_cluster.glob('*.xlsx')):
    df_ = pd.read_excel(f).drop(['prompt_version'], axis='columns')
    # df_agg = df_[df_['cluster_id'] != -1].groupby(['osca_title', 'cluster_id'], as_index=False)['distinct_skill'].apply(lambda x: '\n'.join([f"- {s}" for s in x.head(100)]))
    all_data = df_.to_dict(orient='records')
    osca = df_['OSCA'].values.tolist()[0]
    # if osca not in ['Barista'] : continue
    print(f"----------{osca}-----------")
    counter = 1
    loop = True
    temp = 0.0
    while loop and (counter <= count):
        loop = False
        try:
            sys_msg = get_sys_msg(osca)
            list_ = [item['skill'] for item in all_data]
            list_1 = ['\n'.join([f"- {s}" for s in list_])]
            model_cat = "GPT"
            full_prompts = [instruction_format(sys_msg, "Input Skills: \n" + t, template=model_cat) for t in list_1]

            sampling_params = SamplingParams(
                                            max_tokens=8048,
                                            temperature=temp,
                                            top_p=1, 
                                            frequency_penalty=0,
                                            presence_penalty=0, 
                                            n=1, 
                                            best_of=1
                                            )
            outputs = llm.generate(full_prompts, sampling_params=sampling_params)
            responses = []
            pattern = r"\[([^]]+)\]"
            i = 0
            for out in outputs:
                for output in out.outputs:
                    responses.append(["[" + re.search(pattern, output.text.strip()).group(1) + "]"])
            all_df = []
            for i, res in tqdm(enumerate(responses)):
                json_text = res[0] #res.replace('```', '').replace('json', '')
                df = pd.DataFrame(json.loads(json_text))

                # dict_ = all_data[i]
                # extr_cols = df.columns.tolist()
                # for col in df_.columns:
                #     if col == 'skill': continue
                #     df[col] = dict_[col]
                # df['osca_title'] = osca
                df['prompt_version'] = prompt_v
                all_df.append(df)
            df_res = pd.concat(all_df)
            df_res.reset_index(drop=True, inplace=True)
            df_res = df_.merge(df_res, on='skill')
            # df_res = df_res[df_.columns.tolist() + extr_cols]#.drop(["distinct_skill"], axis='columns')
            print(df_res.shape)
            df_dict = {'Category_Description': df_res}
            write_output_to_excel(df_dict, output_folder_cluster, f"Qual_Skills_Cat_Desc_{"GPT"}_{osca.replace(' ', '_')}", 'cl_5')
        except Exception as e:
            print(f"try {count - counter} more time...{e}, {traceback.format_exc()}")
            loop = True
            if counter <= 5:
                temp += 0.1
            counter += 1

# COMMAND ----------

display(df_res)

# COMMAND ----------

# MAGIC %md
# MAGIC # TEST

# COMMAND ----------

from pathlib import Path
import pandas as pd
input_folder = Path("/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/NST/Input_/ESCO dataset - v1.2.0 - classification - en - csv/")

df_sk = pd.read_csv(input_folder / 'skills_en.csv')
df_occp = pd.read_csv(input_folder / 'occupations_en.csv')
df_lk = pd.read_csv(input_folder / 'occupationSkillRelations_en.csv')
display(df_lk.head(10))

# COMMAND ----------

df_ = df_occp[df_occp['preferredLabel'].str.contains('accountant')][]
df_

# COMMAND ----------

# MAGIC %md
# MAGIC # Playground 1

# COMMAND ----------

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.manifold import TSNE 
from collections import defaultdict, Counter 
import re

col = 'description'
texts = df_[col].tolist()

# === Embed skills ===
task = "text-matching"
embeddings = model_sim.encode(
    texts,
    task=task,
    prompt_name=task,
    show_progress_bar=True
)
embeddings_np = embeddings#.cpu().numpy()

# === Clustering ===
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.7,
    metric='cosine',
    linkage='average'
)
cluster_ids = clustering.fit_predict(embeddings_np)
df_['cluster_id'] = cluster_ids

cluster_groups = defaultdict(list)
for skill, cid in zip(texts, cluster_ids):
    cluster_groups[cid].append(skill)


print(f"Number of clusters: {len(set(cluster_ids))}")

# COMMAND ----------

df_.groupby('cluster_id').count()

# COMMAND ----------

display(df_.head(100))

# COMMAND ----------

skills

# COMMAND ----------

display(df_l[df_l['description_x'] != df_l['description_y']][['description_x', 'description_y', 'Sim']].head(100))

# COMMAND ----------

map_ = {1: {
            "name": "ASC",
            "sheet_name": "Specialist tasks data",
            "cols": ["Specialist Task", "Specialist Cluster"],
            "file": ""
        },
        2: {
            "name": "ONet",
            "sheet_name": "Skills",
            "cols": ["Element Name"],
            "file": ""
        },
        3: {
            "name": "ESCO",
            "sheet_name": None,
            "cols": ["preferredLabel"],
            "file": ""
        }
    }
for i, f in enumerate(list(Path(input_folder).glob('*'))):
    map_[i+1]['file'] = f
map_

# COMMAND ----------

import pandas as pd
df_x = pd.read_excel("/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/NST/Output_01_08_25/Sample_Output_17_2.xlsx", sheet_name="Output")

df_temp = df_x[['noun_based_skill', 'unit_code', 'unit_title','prompt_version']].copy()
df_temp['language'] = 'noun-based'
df_temp.rename({'noun_based_skill': 'skill'}, axis='columns', inplace=True)
df_x.rename({'noun_based_skill': 'language'}, axis='columns', inplace=True)
df_x['language'] = 'Verb-based'
df_x = pd.concat([df_x, df_temp[df_x.columns]])
df_x.drop_duplicates(subset=['skill'], inplace=True)
df_x.reset_index(drop=True, inplace=True)
df_x['skill_code'] = df_x.index + 1
# Create the 5-digit string code
df_x['skill_code'] = df_x['skill_code'].apply(lambda x: f'{x:05d}')
df_x = df_x[['skill_code', 'skill', 'language', 'unit_code', 'unit_title', 'prompt_version']]
display(df_x.head(20))
df_x.shape, df_x.columns

# COMMAND ----------

dict_df = {}
for i in map_:
    item = map_[i]
    x_col = 'skill'
    for y_col in item['cols']:
        if 'xlsx' in item['file'].name:
            df_y = pd.read_excel(item['file'], sheet_name=item['sheet_name'])
        else:
            df_y = pd.read_csv(item['file'])
        df_y.drop_duplicates(subset=[y_col], inplace=True)
        df_y.reset_index(drop=True, inplace=True)
        df_l = get_similarty_by_embedding(df_x, df_y, x_col, y_col, model_sim)
        df_l.rename({'actual_index_x': 'index'}, axis='columns', inplace=True)
        df_l.drop(x_col, axis='columns', inplace=True)
        df_l = df_x.reset_index().merge(df_l, on='index', how='left')
        df_l.drop('index', axis='columns', inplace=True)

        df_l.rename({'actual_index_y': 'index'}, axis='columns', inplace=True)
        df_l.drop(y_col, axis='columns', inplace=True)
        df_l = df_l.merge(df_y.reset_index(), on='index', how='left')
        df_l.drop(['index'], axis='columns', inplace=True)
        df_l.reset_index(inplace=True, drop=True)
        # for col in df_y.columns:
        #     df_l.loc[df_l['Sim'] < 0.5, col] = None
        dict_df[f"{item['name']}_{y_col}"] = df_l
# display(df_l)

# COMMAND ----------

write_output_to_excel(dict_df, output_folder, 'Similarity_Output', prompt_v)

# COMMAND ----------

# MAGIC %md
# MAGIC # Playground

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import gc
import psutil
import os
from collections import Counter
from scipy.sparse import csr_matrix, vstack
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class MemoryOptimizedTextClusterer:
    def __init__(self, language='english', memory_limit_gb=4, batch_size=1000):
        self.language = language
        self.memory_limit_gb = memory_limit_gb
        self.batch_size = batch_size
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                                 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'])
        
        # Results storage
        self.best_model = None
        self.best_score = -1
        self.results = []
        
    def get_memory_usage(self):
        """Monitor current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024  # GB
    
    def preprocess_text_batch(self, texts, use_lemmatization=True, min_word_length=2):
        """Optimized batch text preprocessing"""
        processed = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                processed.append('')
                continue
            
            # Fast preprocessing without heavy regex
            text = str(text).lower()
            
            # Simple cleaning
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                processed.append('')
                continue
            
            # Fast tokenization and filtering
            words = text.split()
            words = [w for w in words if len(w) >= min_word_length and w not in self.stop_words]
            
            # Optional lemmatization (lightweight)
            if use_lemmatization and len(words) < 100:  # Only for shorter texts
                try:
                    words = [self.lemmatizer.lemmatize(w) for w in words]
                except:
                    pass  # Skip if lemmatization fails
            
            processed.append(' '.join(words))
        
        return processed
    
    def create_optimized_vectorizer(self, n_docs, target_features=5000):
        """Create memory-efficient vectorizer based on dataset size"""
        
        # Adaptive feature limits based on dataset size
        if n_docs > 20000:
            max_features = min(target_features, 3000)
            max_df = 0.8
            min_df = max(3, n_docs // 10000)
        elif n_docs > 10000:
            max_features = min(target_features, 5000)
            max_df = 0.85
            min_df = max(2, n_docs // 5000)
        else:
            max_features = target_features
            max_df = 0.9
            min_df = 2
        
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            max_df=max_df,
            min_df=min_df,
            dtype=np.float32,  # Use float32 to save memory
            norm='l2',
            sublinear_tf=True
        )
    
    def incremental_vectorization(self, texts, vectorizer, batch_size=None):
        """Memory-efficient vectorization using batches"""
        if batch_size is None:
            batch_size = self.batch_size
        
        print(f"Vectorizing {len(texts)} documents in batches of {batch_size}...")
        
        # First pass: fit vocabulary on sample
        sample_size = min(10000, len(texts))
        sample_texts = shuffle(texts, random_state=42, n_samples=sample_size)
        vectorizer.fit(sample_texts)
        
        # Second pass: transform all texts in batches
        feature_matrices = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_matrix = vectorizer.transform(batch_texts)
            feature_matrices.append(batch_matrix)
            
            if i % (batch_size * 5) == 0:
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} documents")
                gc.collect()  # Force garbage collection
        
        # Combine all batches
        print("Combining feature matrices...")
        X = vstack(feature_matrices)
        
        # Clean up
        del feature_matrices
        gc.collect()
        
        return X
    
    def smart_dimensionality_reduction(self, X, target_dim=100, method='auto'):
        """Intelligent dimensionality reduction based on data characteristics"""
        n_samples, n_features = X.shape
        
        print(f"Input shape: {n_samples} x {n_features}")
        
        # Skip reduction if already small enough
        if n_features <= target_dim * 2:
            print("Skipping dimensionality reduction - feature space already compact")
            return X.toarray() if hasattr(X, 'toarray') else X, None
        
        # Choose method based on data size and sparsity
        if method == 'auto':
            sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
            
            if sparsity > 0.95 and n_features > 10000:
                method = 'sparse_random'
            elif n_samples > 10000:
                method = 'incremental_pca'
            else:
                method = 'svd'
        
        print(f"Using {method} for dimensionality reduction to {target_dim} dimensions...")
        
        # Apply selected method
        if method == 'sparse_random':
            reducer = SparseRandomProjection(n_components=target_dim, random_state=42)
            X_reduced = reducer.fit_transform(X)
        elif method == 'incremental_pca':
            # Convert to dense in batches for IncrementalPCA
            reducer = IncrementalPCA(n_components=target_dim, batch_size=min(self.batch_size, n_samples//10))
            
            # Process in chunks to avoid memory issues
            chunk_size = min(self.batch_size, n_samples)
            for i in range(0, n_samples, chunk_size):
                chunk = X[i:i+chunk_size].toarray()
                if i == 0:
                    reducer.partial_fit(chunk)
                else:
                    reducer.partial_fit(chunk)
                del chunk
                gc.collect()
            
            # Transform the data
            X_reduced = np.zeros((n_samples, target_dim))
            for i in range(0, n_samples, chunk_size):
                chunk = X[i:i+chunk_size].toarray()
                X_reduced[i:i+chunk_size] = reducer.transform(chunk)
                del chunk
                gc.collect()
        else:  # SVD
            reducer = TruncatedSVD(n_components=target_dim, random_state=42)
            X_reduced = reducer.fit_transform(X)
        
        print(f"Reduced to shape: {X_reduced.shape}")
        return X_reduced, reducer
    
    def estimate_optimal_clusters(self, X, max_k=20, sample_size=20000):
        """Fast cluster number estimation using sampling"""
        n_samples = X.shape[0]
        
        # Use sampling for large datasets
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        print(f"Estimating optimal clusters using {X_sample.shape[0]} samples...")
        
        # Quick estimation using fewer k values
        max_k = min(max_k, X_sample.shape[0] // 10, 15)
        k_values = [x for x in range(2, 100, 2)] #[2, 3, 4, 5, 7, 10, max_k] if max_k > 10 else list(range(2, max_k + 1))
        
        best_score = -1
        best_k = 5
        
        for k in k_values:
            if k >= X_sample.shape[0]:
                break
                
            try:
                # Use MiniBatchKMeans for speed
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=min(self.batch_size, X_sample.shape[0]//2))
                labels = kmeans.fit_predict(X_sample)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_sample, labels, sample_size=min(self.batch_size, X_sample.shape[0]))
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
            except Exception as e:
                print(f"Error with k={k}: {e}")
                continue
        
        print(f"Estimated optimal clusters: {best_k} (score: {best_score:.3f})")
        return best_k
    
    def scalable_clustering(self, X, n_clusters, method='minibatch_kmeans'):
        """Memory-efficient clustering algorithms"""
        print(f"Clustering {X.shape[0]} samples into {n_clusters} clusters using {method}...")
        
        if method == 'minibatch_kmeans':
            # MiniBatchKMeans is much more memory efficient
            batch_size =self.batch_size# min(self.batch_size, X.shape[0] // 10)
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                batch_size=batch_size,
                max_iter=100,
                n_init=3
            )
            labels = clusterer.fit_predict(X)
            
        elif method == 'kmeans' and X.shape[0] < 10000:
            # Regular KMeans only for smaller datasets
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            labels = clusterer.fit_predict(X)
            
        elif method == 'agglomerative' and X.shape[0] < 5000:
            # Agglomerative only for small datasets due to memory complexity
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clusterer.fit_predict(X)
            
        else:
            # Default to MiniBatchKMeans for large datasets
            batch_size = min(1000, X.shape[0] // 10)
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                batch_size=batch_size,
                max_iter=100
            )
            labels = clusterer.fit_predict(X)
        
        return labels, clusterer
    
    def evaluate_clustering_sample(self, X, labels, sample_size=10000):
        """Efficient clustering evaluation using sampling"""
        if len(set(labels)) <= 1:
            return -1, -1
        
        n_samples = X.shape[0]
        
        # Use sampling for large datasets to speed up evaluation
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        try:
            silhouette = silhouette_score(X_sample, labels_sample, sample_size=min(self.batch_size, len(X_sample)))
            calinski = calinski_harabasz_score(X_sample, labels_sample)
            return silhouette, calinski
        except:
            return -1, -1
    
    def fit_predict(self, texts, n_clusters=None, auto_optimize=True):
        """Main clustering method optimized for large datasets"""
        start_memory = self.get_memory_usage()
        print(f"Starting clustering of {len(texts)} documents...")
        print(f"Initial memory usage: {start_memory:.2f} GB")
        
        # Input validation
        if len(texts) < 2:
            print("Error: Need at least 2 documents for clustering")
            return None
        
        # Preprocessing in batches
        print("Preprocessing texts...")
        processed_texts = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            processed_batch = self.preprocess_text_batch(batch)
            processed_texts.extend(processed_batch)
            
            if i % (self.batch_size * 5) == 0:
                current_memory = self.get_memory_usage()
                print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} - Memory: {current_memory:.2f} GB")
                gc.collect()
        
        # CRITICAL FIX: Keep track of which documents are non-empty
        # Create mapping from original indices to processed indices
        valid_docs = []
        original_to_processed = {}
        
        for i, text in enumerate(processed_texts):
            if text.strip():  # Non-empty after processing
                original_to_processed[i] = len(valid_docs)
                valid_docs.append(text)
        
        print(f"Total documents: {len(texts)}")
        print(f"Non-empty documents: {len(valid_docs)}")
        print(f"Empty documents: {len(texts) - len(valid_docs)}")
        
        if len(valid_docs) < 10:
            print("Error: Too few non-empty documents for meaningful clustering")
            return None
        
        # Memory-efficient vectorization on valid documents only
        vectorizer = self.create_optimized_vectorizer(len(valid_docs))
        X = self.incremental_vectorization(valid_docs, vectorizer)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Matrix sparsity: {1.0 - (X.nnz / (X.shape[0] * X.shape[1])):.3f}")
        
        # Dimensionality reduction
        target_dim = 100# min(100, X.shape[1] // 2, len(valid_docs) // 5)
        X_reduced, reducer = self.smart_dimensionality_reduction(X, target_dim=target_dim)
        
        del X  # Free memory
        gc.collect()
        
        # Optimize cluster number if needed
        if auto_optimize and n_clusters is None:
            n_clusters = self.estimate_optimal_clusters(X_reduced)
        elif n_clusters is None:
            n_clusters = min(10, len(valid_docs) // 100, int(np.sqrt(len(valid_docs))))
        
        print(f"Using {n_clusters} clusters")
        
        # Clustering on valid documents only
        cluster_labels, clusterer = self.scalable_clustering(X_reduced, n_clusters)
        
        # Evaluation
        silhouette, calinski = self.evaluate_clustering_sample(X_reduced, cluster_labels)
        
        # CRITICAL FIX: Create output labels array with correct dimensions
        full_labels = np.full(len(texts), -1, dtype=int)  # -1 for empty documents
        
        # Map cluster results back to original positions
        for original_idx, processed_idx in original_to_processed.items():
            if processed_idx < len(cluster_labels):
                full_labels[original_idx] = cluster_labels[processed_idx]
        
        # Store results
        self.best_model = {
            'labels': full_labels,
            'n_clusters': len(set(full_labels)),
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'vectorizer': vectorizer,
            'reducer': reducer,
            'clusterer': clusterer,
            'processed_texts': valid_docs,
            'original_indices': list(original_to_processed.keys()),
            'feature_matrix': X_reduced
        }
        
        # Final validation
        assert len(full_labels) == len(texts), f"Output size {len(full_labels)} != input size {len(texts)}"
        
        final_memory = self.get_memory_usage()
        print(f"\nClustering completed!")
        print(f"Final memory usage: {final_memory:.2f} GB (Peak increase: {final_memory - start_memory:.2f} GB)")
        print(f"Input documents: {len(texts)}")
        print(f"Output labels: {len(full_labels)}")
        print(f"Successfully clustered: {np.sum(full_labels != -1)}")
        print(f"Empty/skipped documents: {np.sum(full_labels == -1)}")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Calinski-Harabasz Score: {calinski:.1f}")
        print(f"Number of Clusters: {len(set(cluster_labels))}")
        
        return full_labels
    
    def get_cluster_keywords(self, texts, n_keywords=5):
        """Memory-efficient keyword extraction"""
        if self.best_model is None:
            return None
        
        cluster_labels = self.best_model['labels']
        processed_texts = self.best_model['processed_texts']
        
        print("Extracting cluster keywords...")
        summaries = {}
        
        # Process each cluster
        for cluster_id in set(cluster_labels):
            cluster_texts = [processed_texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if not cluster_texts:
                continue
            
            # Simple keyword extraction using word frequency
            all_words = []
            for text in cluster_texts[:100]:  # Limit for memory efficiency
                all_words.extend(text.split())
            
            if all_words:
                word_counts = Counter(all_words)
                top_keywords = [word for word, count in word_counts.most_common(n_keywords)]
            else:
                top_keywords = []
            
            summaries[cluster_id] = {
                'size': len(cluster_texts),
                'keywords': top_keywords,
                'sample_texts': cluster_texts[:3]
            }
        
        return summaries
    
    def plot_results_lightweight(self, sample_size=2000):
        """Memory-efficient visualization"""
        if self.best_model is None:
            print("No results to plot")
            return
        
        # Sample data for visualization if too large
        X = self.best_model['feature_matrix']
        labels = self.best_model['labels']
        
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_plot = X[indices]
            labels_plot = labels[indices]
        else:
            X_plot = X
            labels_plot = labels
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D visualization
        if X_plot.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_plot)
        else:
            X_2d = X_plot
        
        scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_plot, 
                                cmap='tab10', alpha=0.6, s=10)
        axes[0].set_title(f'Cluster Visualization ({len(X_plot)} samples)')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        
        # Cluster sizes
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        axes[1].bar(cluster_counts.index, cluster_counts.values)
        axes[1].set_title('Cluster Size Distribution')
        axes[1].set_xlabel('Cluster ID')
        axes[1].set_ylabel('Number of Documents')
        
        plt.tight_layout()
        plt.show()

# Optimized demo function for large datasets
def demo_large_scale_clustering(documents):
    """Demonstrate clustering on large synthetic dataset"""
    
    # print(f"Generating {n_docs} synthetic documents...")
    
    # # Create synthetic documents with clear clusters
    # topics = [
    #     ["machine", "learning", "algorithm", "data", "model", "training", "neural", "network"],
    #     ["sports", "football", "basketball", "player", "game", "team", "match", "score"],
    #     ["cooking", "recipe", "food", "ingredient", "kitchen", "chef", "meal", "dish"],
    #     ["technology", "computer", "software", "programming", "code", "development", "system"],
    #     ["finance", "money", "investment", "stock", "market", "trading", "profit", "bank"],
    #     ["health", "medical", "doctor", "patient", "treatment", "hospital", "medicine", "care"],
    #     ["education", "student", "teacher", "school", "learning", "study", "academic", "knowledge"],
    #     ["travel", "vacation", "tourist", "destination", "journey", "adventure", "culture", "explore"]
    # ]
    
    # documents = []
    # true_labels = []
    
    # docs_per_topic = n_docs // len(topics)
    
    # for topic_idx, words in enumerate(topics):
    #     for _ in range(docs_per_topic):
    #         # Generate document with 10-30 words
    #         doc_length = np.random.randint(10, 31)
    #         doc_words = np.random.choice(words, size=doc_length, replace=True)
            
    #         # Add some random common words
    #         common_words = ["the", "and", "is", "are", "for", "with", "this", "that"]
    #         noise_words = np.random.choice(common_words, size=np.random.randint(3, 8), replace=True)
            
    #         all_words = list(doc_words) + list(noise_words)
    #         np.random.shuffle(all_words)
            
    #         documents.append(" ".join(all_words))
    #         true_labels.append(topic_idx)
    
    # # Add remaining documents to reach exact count
    # remaining = n_docs - len(documents)
    # for _ in range(remaining):
    #     topic_idx = np.random.randint(len(topics))
    #     words = topics[topic_idx]
    #     doc_length = np.random.randint(10, 31)
    #     doc_words = np.random.choice(words, size=doc_length, replace=True)
    #     documents.append(" ".join(doc_words))
    #     true_labels.append(topic_idx)
    
    # # Add some empty documents to test handling
    # empty_count = n_docs // 100  # 1% empty documents
    # for _ in range(empty_count):
    #     documents.append("")
    #     true_labels.append(-1)  # Special label for empty docs
    
    # print(f"Generated {len(documents)} documents across {len(topics)} topics (with {empty_count} empty docs)")
    
    # Initialize clusterer with appropriate settings for large data
    clusterer = MemoryOptimizedTextClusterer(
        memory_limit_gb=12, 
        batch_size=10000  # Larger batches for efficiency
    )
    
    # Perform clustering
    predicted_labels = clusterer.fit_predict(documents, n_clusters=None)
    
    if predicted_labels is not None:
        # Verify dimensions match
        print(f"\nDimension Check:")
        print(f"Input documents: {len(documents)}")
        print(f"Output labels: {len(predicted_labels)}")
        print(f"Dimensions match: {len(documents) == len(predicted_labels)}")
        
        # Only proceed if dimensions match
        if len(documents) == len(predicted_labels):
            # Filter out empty documents for accuracy calculation
            # valid_mask = (predicted_labels != -1) & (np.array(true_labels) != -1)
            # valid_true = np.array(true_labels)[valid_mask]
            # valid_pred = predicted_labels[valid_mask]
            
            # # Calculate accuracy
            # if len(valid_pred) > 0:
            #     from sklearn.metrics import adjusted_rand_score
            #     ari = adjusted_rand_score(valid_true, valid_pred)
            #     print(f"Adjusted Rand Index: {ari:.3f}")
            # else:
            #     print("No valid predictions to evaluate")
            
            # print(f"True clusters: {len(topics)}")
            print(f"Found clusters: {len(set(predicted_labels[predicted_labels != -1]))}")
            
            # Get cluster summaries
            # summaries = clusterer.get_cluster_keywords(documents)
            
            # if summaries:
            #     print("\nCluster Keywords:")
            #     for cluster_id, info in summaries.items():
            #         print(f"Cluster {cluster_id} ({info['size']} docs): {', '.join(info['keywords'][:5])}")
            
            # Lightweight visualization
            clusterer.plot_results_lightweight()
            
            return clusterer
        else:
            print("ERROR: Dimension mismatch between input and output!")
            return None
    else:
        print("Clustering failed!")
        return None

# Run the large-scale demonstration
# if __name__ == "__main__":
#     # Test with 25k documents
#     clusterer = demo_large_scale_clustering(25000)


# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    raise('Can not download the nltk components')
    pass

class AdvancedTextClusterer:
    def __init__(self, language='english'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        self.vectorizers = {}
        self.models = {}
        self.best_model = None
        self.best_score = -1
        self.results = {}
        
    def preprocess_text(self, text, use_lemmatization=True, use_stemming=False, 
                       remove_numbers=True, min_word_length=2):
        """Advanced text preprocessing with multiple options"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove HTML tags and URLs
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers if specified
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stop words and short words
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) >= min_word_length]
            
            # Apply lemmatization or stemming
            if use_lemmatization:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            elif use_stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            return ' '.join(tokens)
        except:
            # Fallback preprocessing without NLTK
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) >= min_word_length]
            return ' '.join(words)
    
    def create_vectorizers(self, max_features_range=[1000, 5000, 10000]):
        """Create multiple vectorizers with different configurations"""
        vectorizers = {}
        
        for max_features in max_features_range:
            # TF-IDF with different n-gram ranges
            vectorizers[f'tfidf_unigram_{max_features}'] = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 1),
                max_df=0.95,
                min_df=2
            )
            
            # vectorizers[f'tfidf_bigram_{max_features}'] = TfidfVectorizer(
            #     max_features=max_features,
            #     ngram_range=(1, 2),
            #     max_df=0.95,
            #     min_df=2
            # )
            
            # # Count vectorizer
            # vectorizers[f'count_{max_features}'] = CountVectorizer(
            #     max_features=max_features,
            #     ngram_range=(1, 2),
            #     max_df=0.95,
            #     min_df=2
            # )
        
        return vectorizers
    
    def apply_dimensionality_reduction(self, X, method='pca', n_components=50):
        """Apply dimensionality reduction"""
        if method == 'pca':
            reducer = PCA(n_components=min(n_components, X.shape[1]-1))
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=min(n_components, X.shape[1]-1))
        elif method == 'lda':
            reducer = LatentDirichletAllocation(n_components=min(n_components, X.shape[1]-1), 
                                              random_state=42, max_iter=10)
        else:
            return X
        
        return reducer.fit_transform(X), reducer
    
    def optimize_clusters(self, X, method='kmeans', max_clusters=15):
        """Find optimal number of clusters"""
        scores = []
        K_range = range(2, min(max_clusters + 1, X.shape[0]))
        
        for k in K_range:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = clusterer.fit_predict(X)
            elif method == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=k)
                labels = clusterer.fit_predict(X)
            else:
                continue
            
            if len(set(labels)) > 1:  # Ensure we have more than 1 cluster
                score = silhouette_score(X, labels)
                scores.append((k, score))
        
        if scores:
            optimal_k = max(scores, key=lambda x: x[1])[0]
            return optimal_k
        return 3  # Default fallback
    
    def ensemble_clustering(self, X, n_clusters):
        """Perform ensemble clustering with multiple algorithms"""
        clustering_results = {}
        
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clustering_results['kmeans'] = kmeans.fit_predict(X)
        
        # Agglomerative Clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        clustering_results['agglomerative'] = agg.fit_predict(X)
        
        # DBSCAN (adaptive eps)
        if X.shape[0] > 100:  # Only for larger datasets
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=max(5, n_clusters))
            neighbors_fit = neighbors.fit(X)
            distances, indices = neighbors_fit.kneighbors(X)
            distances = np.sort(distances[:, -1], axis=0)
            eps = distances[int(0.9 * len(distances))]  # 90th percentile
            
            dbscan = DBSCAN(eps=eps, min_samples=max(3, n_clusters//2))
            dbscan_labels = dbscan.fit_predict(X)
            
            if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
                clustering_results['dbscan'] = dbscan_labels
        
        return clustering_results
    
    def evaluate_clustering(self, X, labels, method_name):
        """Comprehensive clustering evaluation"""
        if len(set(labels)) <= 1:
            return {'method': method_name, 'silhouette': -1, 'calinski_harabasz': -1, 'n_clusters': len(set(labels))}
        
        try:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            
            return {
                'method': method_name,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'n_clusters': len(set(labels)),
                'labels': labels
            }
        except:
            return {'method': method_name, 'silhouette': -1, 'calinski_harabasz': -1, 'n_clusters': len(set(labels))}
    
    def fit_predict(self, texts, n_clusters=None, auto_optimize=True):
        """Main method to perform advanced text clustering"""
        print("Starting advanced text clustering...")
        
        # Preprocess texts
        print("Preprocessing texts...")
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Filter out empty texts
        non_empty_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
        processed_texts = [processed_texts[i] for i in non_empty_indices]
        
        if len(processed_texts) < 2:
            print("Error: Need at least 2 non-empty texts for clustering")
            return None
        
        # Create vectorizers
        print("Creating vectorizers...")
        vectorizers = self.create_vectorizers()
        
        best_overall_score = -1
        best_result = None
        all_results = []
        
        # Try different vectorization and clustering combinations
        for vec_name, vectorizer in vectorizers.items():
            print(f"Processing with {vec_name}...")
            
            try:
                # Vectorize texts
                X = vectorizer.fit_transform(processed_texts)
                
                # Skip if matrix is too sparse or small
                if X.shape[1] < 10 or X.nnz < 10:
                    continue
                
                # Apply dimensionality reduction
                X_reduced, reducer = self.apply_dimensionality_reduction(X.toarray(), method='pca')
                
                # Optimize number of clusters if not specified
                if auto_optimize and n_clusters is None:
                    optimal_k = self.optimize_clusters(X_reduced, method='kmeans')
                else:
                    optimal_k = n_clusters or min(5, len(processed_texts)//2)
                
                # Perform ensemble clustering
                clustering_results = self.ensemble_clustering(X_reduced, optimal_k)
                
                # Evaluate each clustering method
                for method_name, labels in clustering_results.items():
                    result = self.evaluate_clustering(X_reduced, labels, f"{vec_name}_{method_name}")
                    result['vectorizer'] = vec_name
                    result['clustering_method'] = method_name
                    result['optimal_k'] = optimal_k
                    
                    all_results.append(result)
                    
                    # Track best result
                    if result['silhouette'] > best_overall_score:
                        best_overall_score = result['silhouette']
                        best_result = result.copy()
                        best_result['vectorizer_obj'] = vectorizer
                        best_result['reducer'] = reducer
                        best_result['feature_matrix'] = X_reduced
                        best_result['processed_texts'] = processed_texts
                        best_result['original_indices'] = non_empty_indices
            
            except Exception as e:
                print(f"Error processing {vec_name}: {str(e)}")
                continue
        
        # Store results
        self.results = pd.DataFrame(all_results)
        self.best_model = best_result
        
        if best_result is not None:
            print(f"\nBest clustering result:")
            print(f"Method: {best_result['vectorizer']}_{best_result['clustering_method']}")
            print(f"Silhouette Score: {best_result['silhouette']:.3f}")
            print(f"Calinski-Harabasz Score: {best_result['calinski_harabasz']:.3f}")
            print(f"Number of Clusters: {best_result['n_clusters']}")
            
            # Create full labels array (including empty texts)
            full_labels = np.full(len(texts), -1)  # -1 for empty texts
            full_labels[best_result['original_indices']] = best_result['labels']
            
            return full_labels
        else:
            print("No valid clustering results found.")
            return None
    
    def plot_results(self, figsize=(15, 10)):
        """Visualize clustering results"""
        if self.best_model is None:
            print("No clustering results to plot. Run fit_predict first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cluster visualization (2D PCA)
        X_2d = PCA(n_components=2).fit_transform(self.best_model['feature_matrix'])
        scatter = axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], 
                                   c=self.best_model['labels'], 
                                   cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('Cluster Visualization (PCA)')
        axes[0, 0].set_xlabel('First Principal Component')
        axes[0, 0].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Silhouette scores comparison
        if hasattr(self, 'results') and not self.results.empty:
            top_results = self.results.nlargest(10, 'silhouette')
            axes[0, 1].barh(range(len(top_results)), top_results['silhouette'])
            axes[0, 1].set_yticks(range(len(top_results)))
            axes[0, 1].set_yticklabels([f"{row['vectorizer']}_{row['clustering_method']}" 
                                       for _, row in top_results.iterrows()], fontsize=8)
            axes[0, 1].set_title('Top 10 Silhouette Scores')
            axes[0, 1].set_xlabel('Silhouette Score')
        
        # 3. Cluster size distribution
        cluster_counts = pd.Series(self.best_model['labels']).value_counts().sort_index()
        axes[1, 0].bar(cluster_counts.index, cluster_counts.values)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Documents')
        
        # 4. Method comparison heatmap
        if hasattr(self, 'results') and not self.results.empty:
            pivot_data = self.results.pivot_table(
                index='clustering_method', 
                columns='vectorizer', 
                values='silhouette', 
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 1])
            axes[1, 1].set_title('Method Performance Heatmap')
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_summaries(self, texts, n_keywords=5):
        """Get keyword summaries for each cluster"""
        if self.best_model is None:
            print("No clustering results available.")
            return None
        
        labels = self.best_model['labels']
        processed_texts = self.best_model['processed_texts']
        
        summaries = {}
        
        # Create TF-IDF for keyword extraction
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_texts = [processed_texts[i] for i, label in enumerate(labels) if label == cluster_id]
            cluster_size = len(cluster_texts)
            
            if cluster_texts:
                # Get top keywords for this cluster
                try:
                    tfidf_matrix = tfidf.fit_transform(cluster_texts)
                    feature_names = tfidf.get_feature_names_out()
                    
                    # Calculate mean TF-IDF scores
                    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    top_indices = mean_scores.argsort()[-n_keywords:][::-1]
                    top_keywords = [feature_names[i] for i in top_indices]
                    
                    summaries[cluster_id] = {
                        'size': cluster_size,
                        'keywords': top_keywords,
                        'sample_texts': cluster_texts[:3]  # First 3 texts as samples
                    }
                except:
                    summaries[cluster_id] = {
                        'size': cluster_size,
                        'keywords': [],
                        'sample_texts': cluster_texts[:3]
                    }
        
        return summaries

# Example usage and demonstration
def advanced_clustering(sample_texts):
    """Demonstrate the advanced text clustering system"""
    
    # Initialize and run clustering
    clusterer = AdvancedTextClusterer()
    labels = clusterer.fit_predict(sample_texts, auto_optimize=True)
    
    if labels is not None:
        # Display results
        # print("\n" + "="*60)
        # print("CLUSTERING RESULTS")
        # print("="*60)
        
        # Show cluster assignments
        # for i, (text, label) in enumerate(zip(sample_texts, labels)):
        #     print(f"Text {i+1} (Cluster {label}): {text[:50]}...")
        
        # Get cluster summaries
        # summaries = clusterer.get_cluster_summaries(sample_texts)
        
        # print("\n" + "="*60)
        # print("CLUSTER SUMMARIES")
        # print("="*60)
        
        # for cluster_id, summary in summaries.items():
        #     print(f"\nCluster {cluster_id} ({summary['size']} documents):")
        #     print(f"Keywords: {', '.join(summary['keywords'])}")
        #     print("Sample texts:")
        #     for i, text in enumerate(summary['sample_texts'], 1):
        #         print(f"  {i}. {text[:60]}...")
        
        # Plot results
        # clusterer.plot_results()
        
        return clusterer
    else:
        print("Clustering failed. Please check your input data.")
        return None
   


# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import gc
import psutil
import os
from collections import Counter
from scipy.sparse import csr_matrix, vstack
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers - the best embedding model
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ SentenceTransformers available - using state-of-the-art embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  SentenceTransformers not available. Install with: pip install sentence-transformers")
    print("   Falling back to TF-IDF for now, but embeddings are recommended for better results")

# Try to import transformers for additional models
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available for additional embedding options")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: pip install transformers torch")

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class AdvancedTextClusterer:
    def __init__(self, language='english', memory_limit_gb=4, batch_size=1000, 
                 embedding_model='sentence-transformers', model_name='auto'):
        self.language = language
        self.memory_limit_gb = memory_limit_gb
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.model_name = model_name
        
        # Initialize embedding model
        self.embedder = None
        self.embedding_dim = None
        self._initialize_embedding_model()
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                                 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'])
        
        # Results storage
        self.best_model = None
        self.best_score = -1
        self.results = []
    
    def _initialize_embedding_model(self):
        """Initialize the best available embedding model"""
        
        if self.embedding_model == 'sentence-transformers' and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Choose the best model based on use case
                if self.model_name == 'auto':
                    # Select best model based on task
                    model_options = [
                        "jinaai/jina-embeddings-v3",
                        '/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/Models/all-MiniLM-L6-v2/',        # Fast, good general purpose (384 dim)
                        'all-mpnet-base-v2',       # Best quality (768 dim)
                        'paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual
                    ]
                    
                    # Try models in order of preference
                    for model_name in model_options:
                        try:
                            print(f"Loading SentenceTransformer model: {model_name}")
                            self.embedder = SentenceTransformer(model_name, trust_remote_code=True)
                            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
                            self.model_name = model_name
                            print(f"✅ Successfully loaded {model_name} (dim: {self.embedding_dim})")
                            return
                        except Exception as e:
                            print(f"Failed to load {model_name}: {e}")
                            continue
                else:
                    # Use specified model
                    print(f"Loading specified SentenceTransformer model: {self.model_name}")
                    self.embedder = SentenceTransformer(self.model_name)
                    self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
                    print(f"✅ Successfully loaded {self.model_name} (dim: {self.embedding_dim})")
                    return
                    
            except Exception as e:
                print(f"Failed to initialize SentenceTransformer: {e}")
                self.embedder = None
        
        elif self.embedding_model == 'transformers' and TRANSFORMERS_AVAILABLE:
            try:
                # Use transformers library
                if self.model_name == 'auto':
                    self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                
                print(f"Loading Transformers model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.transformer_model = AutoModel.from_pretrained(self.model_name)
                self.embedding_dim = self.transformer_model.config.hidden_size
                print(f"✅ Successfully loaded {self.model_name} (dim: {self.embedding_dim})")
                return
                
            except Exception as e:
                print(f"Failed to initialize Transformers model: {e}")
                self.embedder = None
        
        # Fallback to TF-IDF if embeddings fail
        if self.embedder is None and self.embedding_model != 'tfidf':
            print("⚠️  Falling back to TF-IDF vectorization")
            self.embedding_model = 'tfidf'
            self.embedding_dim = None
    
    def get_memory_usage(self):
        """Monitor current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024 / 1024  # GB
        except:
            return 0.0
    
    def preprocess_text_batch(self, texts, use_lemmatization=True, min_word_length=2):
        """Optimized batch text preprocessing"""
        processed = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                processed.append('')
                continue
            
            # Fast preprocessing without heavy regex
            text = str(text).lower()
            
            # Simple cleaning
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                processed.append('')
                continue
            
            # Fast tokenization and filtering
            words = text.split()
            words = [w for w in words if len(w) >= min_word_length and w not in self.stop_words]
            
            # Optional lemmatization (lightweight)
            if use_lemmatization and len(words) < 100:  # Only for shorter texts
                try:
                    words = [self.lemmatizer.lemmatize(w) for w in words]
                except:
                    pass  # Skip if lemmatization fails
            
            processed.append(' '.join(words))
        
        return processed
    
    def create_embeddings(self, texts, batch_size=None):
        """Create embeddings using the best available model"""
        if batch_size is None:
            batch_size = min(self.batch_size, 32)  # Smaller batches for GPU memory
        
        print(f"Creating embeddings for {len(texts)} documents using {self.embedding_model}...")
        
        if self.embedding_model == 'sentence-transformers' and self.embedder is not None:
            return self._create_sentence_transformer_embeddings(texts, batch_size)
        elif self.embedding_model == 'transformers' and hasattr(self, 'transformer_model'):
            return self._create_transformer_embeddings(texts, batch_size)
        else:
            # Fallback to TF-IDF
            return self._create_tfidf_embeddings(texts)
    
    def _create_sentence_transformer_embeddings(self, texts, batch_size):
        """Create embeddings using SentenceTransformers (recommended)"""
        embeddings = []
        
        print(f"Using SentenceTransformer: {self.model_name}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                if 'ninja' in self.model_name:
                    task = "text-matching"
                    batch_embeddings = self.embedder.encode(
                        texts,
                        task=task,
                        prompt_name=task,
                        batch_size=min(batch_size, 32),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # L2 normalization for better clustering
                        )
                else:
                    batch_embeddings = self.embedder.encode(
                        batch_texts,
                        batch_size=min(batch_size, 32),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # L2 normalization for better clustering
                    )
                embeddings.append(batch_embeddings)
                
                if i % (batch_size * 10) == 0:
                    current_memory = self.get_memory_usage()
                    print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} - Memory: {current_memory:.2f} GB")
                    gc.collect()
                    
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Create zero embeddings for failed batch
                batch_embeddings = np.zeros((len(batch_texts), self.embedding_dim))
                embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        final_embeddings = np.vstack(embeddings)
        print(f"Created embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def _create_transformer_embeddings(self, texts, batch_size):
        """Create embeddings using Transformers library"""
        embeddings = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer_model.to(device)
        self.transformer_model.eval()
        
        print(f"Using Transformers model on {device}: {self.model_name}")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(device)
                    
                    # Get embeddings
                    outputs = self.transformer_model(**inputs)
                    
                    # Use mean pooling of last hidden states
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Normalize and convert to numpy
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    embeddings.append(batch_embeddings.cpu().numpy())
                    
                    if i % (batch_size * 10) == 0:
                        current_memory = self.get_memory_usage()
                        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} - Memory: {current_memory:.2f} GB")
                        
                except Exception as e:
                    print(f"Error in batch {i//batch_size}: {e}")
                    batch_embeddings = np.zeros((len(batch_texts), self.embedding_dim))
                    embeddings.append(batch_embeddings)
        
        final_embeddings = np.vstack(embeddings)
        print(f"Created embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def _create_tfidf_embeddings(self, texts):
        """Fallback TF-IDF embeddings"""
        print("Using TF-IDF as fallback...")
        
        # Adaptive feature limits based on dataset size
        n_docs = len(texts)
        if n_docs > 20000:
            max_features = 3000
            max_df = 0.8
            min_df = max(3, n_docs // 10000)
        elif n_docs > 10000:
            max_features = 5000
            max_df = 0.85
            min_df = max(2, n_docs // 5000)
        else:
            max_features = 5000
            max_df = 0.9
            min_df = 2
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            max_df=max_df,
            min_df=min_df,
            dtype=np.float32,
            norm='l2',
            sublinear_tf=True
        )
        
        # Process in batches for large datasets
        if n_docs > 5000:
            # Fit vocabulary on sample
            sample_size = min(5000, n_docs)
            sample_texts = shuffle(texts, random_state=42, n_samples=sample_size)
            vectorizer.fit(sample_texts)
            
            # Transform in batches
            feature_matrices = []
            batch_size = self.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_matrix = vectorizer.transform(batch_texts)
                feature_matrices.append(batch_matrix)
                
                if i % (batch_size * 5) == 0:
                    print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} documents")
                    gc.collect()
            
            # Combine all batches
            X = vstack(feature_matrices)
            del feature_matrices
            gc.collect()
        else:
            X = vectorizer.fit_transform(texts)
        
        return X.toarray(), vectorizer
    
    def smart_dimensionality_reduction(self, X, target_dim=100):
        """Intelligent dimensionality reduction - optimized for embeddings"""
        n_samples, n_features = X.shape
        
        print(f"Input shape: {n_samples} x {n_features}")
        
        # For embeddings, we often don't need as much reduction
        if self.embedding_model in ['sentence-transformers', 'transformers']:
            # Embeddings are already dense and well-structured
            if n_features <= 7680:  # Most embedding models are 384-768 dim
                print("Skipping dimensionality reduction - embeddings already optimal size")
                return X, None
            else:
                # Only reduce if very high dimensional
                target_dim = min(target_dim, n_features // 2)
        
        # Skip reduction if already small enough
        if n_features <= target_dim * 2:
            print("Skipping dimensionality reduction - feature space already compact")
            return X, None
        
        # Choose method based on data size
        if n_samples > 10000:
            print(f"Using IncrementalPCA for dimensionality reduction to {target_dim} dimensions...")
            reducer = IncrementalPCA(n_components=target_dim, batch_size=min(1000, n_samples//10))
            
            # Process in chunks to avoid memory issues
            chunk_size = min(2000, n_samples)
            for i in range(0, n_samples, chunk_size):
                chunk = X[i:i+chunk_size]
                reducer.partial_fit(chunk)
                gc.collect()
            
            X_reduced = reducer.transform(X)
        else:
            print(f"Using PCA for dimensionality reduction to {target_dim} dimensions...")
            reducer = PCA(n_components=target_dim, random_state=42)
            X_reduced = reducer.fit_transform(X)
        
        print(f"Reduced to shape: {X_reduced.shape}")
        return X_reduced, reducer
    
    def estimate_optimal_clusters(self, X, max_k=20, sample_size=20000):
        """Fast cluster number estimation using sampling"""
        n_samples = X.shape[0]
        
        # Use sampling for large datasets
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        print(f"Estimating optimal clusters using {X_sample.shape[0]} samples...")
        
        # Quick estimation using fewer k values
        max_k = min(max_k, X_sample.shape[0] // 10, 15)
        k_values = [x for x in range(2, 100, 2)]#[2, 3, 4, 5, 7, 10, max_k] if max_k > 10 else list(range(2, max_k + 1))
        
        best_score = -1
        best_k = 5
        
        for k in k_values:
            if k >= X_sample.shape[0]:
                break
                
            try:
                # Use MiniBatchKMeans for speed
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=min(self.batch_size, X_sample.shape[0]//2))
                labels = kmeans.fit_predict(X_sample)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_sample, labels, sample_size=min(self.batch_size, X_sample.shape[0]))
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
            except Exception as e:
                print(f"Error with k={k}: {e}")
                continue
        
        print(f"Estimated optimal clusters: {best_k} (score: {best_score:.3f})")
        return best_k
    
    def scalable_clustering(self, X, n_clusters):
        """Memory-efficient clustering algorithms"""
        print(f"Clustering {X.shape[0]} samples into {n_clusters} clusters...")
        
        if X.shape[0] > 10000:
            # Use MiniBatchKMeans for large datasets
            batch_size = min(self.batch_size, X.shape[0] // 10)
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                batch_size=batch_size,
                max_iter=100,
                n_init=3
            )
        else:
            # Use regular KMeans for smaller datasets
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        
        labels = clusterer.fit_predict(X)
        return labels, clusterer
    
    def evaluate_clustering_sample(self, X, labels, sample_size=20000):
        """Efficient clustering evaluation using sampling"""
        if len(set(labels)) <= 1:
            return -1, -1
        
        n_samples = X.shape[0]
        
        # Use sampling for large datasets to speed up evaluation
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        try:
            silhouette = silhouette_score(X_sample, labels_sample, sample_size=min(1000, len(X_sample)))
            calinski = calinski_harabasz_score(X_sample, labels_sample)
            return silhouette, calinski
        except:
            return -1, -1
    
    def fit_predict(self, texts, n_clusters=None, auto_optimize=True):
        """Main clustering method optimized for large datasets"""
        start_memory = self.get_memory_usage()
        print(f"Starting clustering of {len(texts)} documents...")
        print(f"Initial memory usage: {start_memory:.2f} GB")
        
        # Input validation
        if len(texts) < 2:
            print("Error: Need at least 2 documents for clustering")
            return None
        
        # Preprocessing in batches
        print("Preprocessing texts...")
        processed_texts = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            processed_batch = self.preprocess_text_batch(batch)
            processed_texts.extend(processed_batch)
            
            if i % (self.batch_size * 5) == 0:
                current_memory = self.get_memory_usage()
                print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} - Memory: {current_memory:.2f} GB")
                gc.collect()
        
        # Keep track of which documents are non-empty
        valid_docs = []
        original_to_processed = {}
        
        for i, text in enumerate(processed_texts):
            if text.strip():  # Non-empty after processing
                original_to_processed[i] = len(valid_docs)
                valid_docs.append(text)
        
        print(f"Total documents: {len(texts)}")
        print(f"Non-empty documents: {len(valid_docs)}")
        print(f"Empty documents: {len(texts) - len(valid_docs)}")
        
        if len(valid_docs) < 10:
            print("Error: Too few non-empty documents for meaningful clustering")
            return None
        
        # Create embeddings or TF-IDF features
        X_embeddings = self.create_embeddings(valid_docs)
        
        print(f"Feature matrix shape: {X_embeddings.shape}")
        
        # Dimensionality reduction (adaptive based on embedding type)
        if self.embedding_model in ['sentence-transformers', 'transformers']:
            # For embeddings, use smaller target dimension or skip reduction
            target_dim = min(100, X_embeddings.shape[1] // 2)
        else:
            # For TF-IDF, more aggressive reduction may be needed
            target_dim = min(100, X_embeddings.shape[1] // 2, len(valid_docs) // 5)
            
        X_reduced, reducer = self.smart_dimensionality_reduction(X_embeddings, target_dim=target_dim)
        
        del X_embeddings  # Free memory
        gc.collect()
        
        # Optimize cluster number if needed
        if auto_optimize and n_clusters is None:
            n_clusters = self.estimate_optimal_clusters(X_reduced)
        elif n_clusters is None:
            n_clusters = min(10, len(valid_docs) // 100, int(np.sqrt(len(valid_docs))))
        
        print(f"Using {n_clusters} clusters")
        
        # Clustering on valid documents only
        cluster_labels, clusterer = self.scalable_clustering(X_reduced, n_clusters)
        
        # Evaluation
        silhouette, calinski = self.evaluate_clustering_sample(X_reduced, cluster_labels)
        
        # Create output labels array with correct dimensions
        full_labels = np.full(len(texts), -1, dtype=int)  # -1 for empty documents
        
        # Map cluster results back to original positions
        for original_idx, processed_idx in original_to_processed.items():
            if processed_idx < len(cluster_labels):
                full_labels[original_idx] = cluster_labels[processed_idx]
        
        # Store results
        self.best_model = {
            'labels': full_labels,
            'n_clusters': len(set(full_labels)),
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'clusterer': clusterer,
            'processed_texts': valid_docs,
            'original_indices': list(original_to_processed.keys()),
            'feature_matrix': X_reduced,
            'embedding_model': self.embedding_model,
            'model_name': getattr(self, 'model_name', 'unknown')
        }
        
        # Final validation
        assert len(full_labels) == len(texts), f"Output size {len(full_labels)} != input size {len(texts)}"
        
        final_memory = self.get_memory_usage()
        print(f"\nClustering completed!")
        print(f"Final memory usage: {final_memory:.2f} GB (Peak increase: {final_memory - start_memory:.2f} GB)")
        print(f"Input documents: {len(texts)}")
        print(f"Output labels: {len(full_labels)}")
        print(f"Successfully clustered: {np.sum(full_labels != -1)}")
        print(f"Empty/skipped documents: {np.sum(full_labels == -1)}")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Calinski-Harabasz Score: {calinski:.1f}")
        print(f"Number of Clusters: {len(set(cluster_labels))}")
        
        return full_labels
    
    def generate_cluster_descriptions(self, original_texts, use_tfidf=True, max_description_length=200):
        """Generate meaningful descriptions for each cluster using multiple techniques"""
        if self.best_model is None:
            print("No clustering results available. Run fit_predict first.")
            return None
        
        print("Generating comprehensive cluster descriptions...")
        
        # Get the full labels array (includes -1 for empty docs)
        full_labels = np.full(len(original_texts), -1, dtype=int)
        cluster_labels = self.best_model['labels']
        original_indices = self.best_model['original_indices']
        
        # Map cluster labels back to original positions
        for i, original_idx in enumerate(original_indices):
            if i < len(cluster_labels):
                full_labels[original_idx] = cluster_labels[i]
        
        descriptions = {}
        
        # Process each cluster
        unique_clusters = set(cluster_labels)
        for cluster_id in unique_clusters:
            print(f"Processing cluster {cluster_id}...")
            
            # Get original texts for this cluster
            cluster_mask = (full_labels == cluster_id)
            cluster_texts = [original_texts[i] for i in range(len(original_texts)) if cluster_mask[i]]
            
            if not cluster_texts:
                continue
            
            # Method 1: TF-IDF based keyword extraction
            if use_tfidf and len(cluster_texts) > 1:
                distinctive_keywords = self._extract_tfidf_keywords(cluster_texts, n_keywords=10)
            else:
                distinctive_keywords = self._extract_frequency_keywords(cluster_texts, n_keywords=10)
            
            # Method 2: N-gram analysis for phrases
            common_phrases = self._extract_common_phrases(cluster_texts, n_phrases=5)
            
            # Method 3: Topic themes identification
            themes = self._identify_themes(cluster_texts, distinctive_keywords)
            
            # Method 4: Statistical summary
            stats = self._generate_cluster_stats(cluster_texts)
            
            # Method 5: Generate natural language description
            description = self._create_natural_description(
                cluster_id, distinctive_keywords, common_phrases, themes, stats, max_description_length
            )
            
            descriptions[cluster_id] = {
                'description': description,
                'keywords': distinctive_keywords,
                'phrases': common_phrases,
                'themes': themes,
                'statistics': stats,
                'sample_texts': cluster_texts[:3],
                'size': len(cluster_texts)
            }
        
        return descriptions
    
    def _extract_tfidf_keywords(self, texts, n_keywords=10):
        """Extract distinctive keywords using TF-IDF"""
        try:
            # Create TF-IDF vectorizer for this cluster
            vectorizer = TfidfVectorizer(
                max_features=min(1000, len(texts) * 10),
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=1,
                stop_words=list(self.stop_words)
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = mean_scores.argsort()[-n_keywords:][::-1]
            keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            return keywords
        except:
            return self._extract_frequency_keywords(texts, n_keywords)
    
    def _extract_frequency_keywords(self, texts, n_keywords=10):
        """Extract keywords using simple frequency analysis"""
        all_words = []
        for text in texts:
            words = str(text).lower().split()
            words = [w for w in words if len(w) > 2 and w not in self.stop_words]
            all_words.extend(words)
        
        if not all_words:
            return []
        
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Calculate relative frequency
        keywords = [(word, count/total_words) for word, count in word_counts.most_common(n_keywords)]
        return keywords
    
    def _extract_common_phrases(self, texts, n_phrases=5):
        """Extract common phrases (2-3 word combinations)"""
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Extract 2-3 word phrases
            phrase_vectorizer = CountVectorizer(
                ngram_range=(2, 3),
                max_features=200,
                max_df=0.8,
                min_df=2,
                stop_words=list(self.stop_words)
            )
            
            phrase_matrix = phrase_vectorizer.fit_transform(texts)
            feature_names = phrase_vectorizer.get_feature_names_out()
            
            # Get phrase frequencies
            phrase_counts = np.sum(phrase_matrix.toarray(), axis=0)
            
            # Get top phrases
            top_indices = phrase_counts.argsort()[-n_phrases:][::-1]
            phrases = [(feature_names[i], phrase_counts[i]) for i in top_indices if phrase_counts[i] > 1]
            
            return phrases
        except:
            return []
    
    def _identify_themes(self, texts, keywords):
        """Identify main themes based on keyword patterns"""
        if not keywords:
            return []
        
        # Predefined theme categories (expandable)
        theme_patterns = {
            'Technology': ['technology', 'computer', 'software', 'digital', 'system', 'data', 'algorithm', 'programming', 'code', 'internet', 'online', 'tech'],
            'Business': ['business', 'company', 'market', 'sales', 'revenue', 'profit', 'customer', 'service', 'management', 'strategy', 'finance', 'investment'],
            'Health': ['health', 'medical', 'doctor', 'patient', 'treatment', 'hospital', 'medicine', 'care', 'disease', 'therapy', 'wellness', 'fitness'],
            'Education': ['education', 'school', 'student', 'teacher', 'learning', 'study', 'academic', 'university', 'knowledge', 'research', 'training'],
            'Sports': ['sports', 'game', 'team', 'player', 'match', 'score', 'win', 'competition', 'athletic', 'football', 'basketball', 'soccer'],
            'Food': ['food', 'cooking', 'recipe', 'restaurant', 'meal', 'dish', 'ingredient', 'kitchen', 'chef', 'eat', 'dining', 'cuisine'],
            'Travel': ['travel', 'trip', 'vacation', 'tourist', 'destination', 'journey', 'hotel', 'flight', 'adventure', 'culture', 'visit'],
            'Entertainment': ['movie', 'music', 'show', 'entertainment', 'film', 'actor', 'artist', 'performance', 'media', 'celebrity', 'fun'],
            'Science': ['science', 'research', 'study', 'experiment', 'discovery', 'analysis', 'theory', 'method', 'result', 'evidence'],
            'Politics': ['political', 'government', 'policy', 'election', 'vote', 'democracy', 'law', 'legislation', 'official', 'public']
        }
        
        # Extract keyword words only (remove scores)
        keyword_words = [kw[0] if isinstance(kw, tuple) else kw for kw in keywords]
        keyword_text = ' '.join(keyword_words).lower()
        
        # Score themes based on keyword matches
        theme_scores = {}
        for theme, pattern_words in theme_patterns.items():
            score = sum(1 for word in pattern_words if word in keyword_text)
            if score > 0:
                theme_scores[theme] = score
        
        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes[:3] if score >= 1]
    
    def _generate_cluster_stats(self, texts):
        """Generate statistical summary of the cluster"""
        if not texts:
            return {}
        
        text_lengths = [len(str(text).split()) for text in texts]
        
        stats = {
            'document_count': len(texts),
            'avg_document_length': np.mean(text_lengths),
            'min_document_length': np.min(text_lengths),
            'max_document_length': np.max(text_lengths),
            'total_words': sum(text_lengths)
        }
        
        return stats
    
    def _create_natural_description(self, cluster_id, keywords, phrases, themes, stats, max_length=200):
        """Create a natural language description of the cluster"""
        description_parts = []
        
        # Start with cluster size
        size = stats.get('document_count', 0)
        description_parts.append(f"This cluster contains {size} documents")
        
        # Add themes if identified
        if themes:
            if len(themes) == 1:
                description_parts.append(f"primarily focused on {themes[0].lower()}")
            elif len(themes) == 2:
                description_parts.append(f"covering {themes[0].lower()} and {themes[1].lower()}")
            else:
                description_parts.append(f"spanning {', '.join(themes[:-1]).lower()}, and {themes[-1].lower()}")
        
        # Add key concepts
        if keywords:
            top_keywords = [kw[0] if isinstance(kw, tuple) else kw for kw in keywords[:5]]
            description_parts.append(f"Key concepts include: {', '.join(top_keywords)}")
        
        # Add common phrases if available
        if phrases:
            top_phrases = [phrase[0] if isinstance(phrase, tuple) else phrase for phrase in phrases[:3]]
            description_parts.append(f"Common phrases: '{', '.join(top_phrases)}'")
        
        # Add document characteristics
        avg_length = stats.get('avg_document_length', 0)
        if avg_length > 0:
            if avg_length < 10:
                length_desc = "short"
            elif avg_length < 50:
                length_desc = "medium-length"
            else:
                length_desc = "long"
            description_parts.append(f"Documents are typically {length_desc} ({avg_length:.1f} words on average)")
        
        # Combine and truncate if necessary
        full_description = ". ".join(description_parts) + "."
        
        if len(full_description) > max_length:
            # Truncate at sentence boundary
            sentences = full_description.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + ". ") <= max_length:
                    truncated += sentence + ". "
                else:
                    break
            full_description = truncated.rstrip()
        
        return full_description
    
    def get_cluster_keywords(self, texts, n_keywords=5):
        """Simple keyword extraction for quick overview"""
        if self.best_model is None:
            return None
        
        cluster_labels = self.best_model['labels']
        processed_texts = self.best_model['processed_texts']
        
        print("Extracting cluster keywords...")
        summaries = {}
        
        # Process each cluster
        for cluster_id in set(cluster_labels):
            cluster_texts = [processed_texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if not cluster_texts:
                continue
            
            # Simple keyword extraction using word frequency
            all_words = []
            for text in cluster_texts[:100]:  # Limit for memory efficiency
                all_words.extend(text.split())
            
            if all_words:
                word_counts = Counter(all_words)
                top_keywords = [word for word, count in word_counts.most_common(n_keywords)]
            else:
                top_keywords = []
            
            summaries[cluster_id] = {
                'size': len(cluster_texts),
                'keywords': top_keywords,
                'sample_texts': cluster_texts[:3]
            }
        
        return summaries
    
    def plot_results_lightweight(self, sample_size=2000):
        """Memory-efficient visualization"""
        if self.best_model is None:
            print("No results to plot")
            return
        
        # Sample data for visualization if too large
        X = self.best_model['feature_matrix']
        labels = self.best_model['labels']
        
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_plot = X[indices]
            labels_plot = labels[indices]
        else:
            X_plot = X
            labels_plot = labels
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D visualization
        if X_plot.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_plot)
        else:
            X_2d = X_plot
        
        scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_plot, 
                                cmap='tab10', alpha=0.6, s=10)
        axes[0].set_title(f'Cluster Visualization ({len(X_plot)} samples)')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        
        # Cluster sizes
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        axes[1].bar(cluster_counts.index, cluster_counts.values)
        axes[1].set_title('Cluster Size Distribution')
        axes[1].set_xlabel('Cluster ID')
        axes[1].set_ylabel('Number of Documents')
        
        plt.tight_layout()
        plt.show()

# Demo function for large datasets with embeddings
def demo_large_scale_clustering(documents):
    """Demonstrate clustering on large synthetic dataset using embeddings"""
    
    # print(f"Generating {n_docs} synthetic documents...")
    
    # # Create synthetic documents with clear clusters
    # topics = [
    #     ["machine", "learning", "algorithm", "data", "model", "training", "neural", "network"],
    #     ["sports", "football", "basketball", "player", "game", "team", "match", "score"],
    #     ["cooking", "recipe", "food", "ingredient", "kitchen", "chef", "meal", "dish"],
    #     ["technology", "computer", "software", "programming", "code", "development", "system"],
    #     ["finance", "money", "investment", "stock", "market", "trading", "profit", "bank"],
    #     ["health", "medical", "doctor", "patient", "treatment", "hospital", "medicine", "care"],
    #     ["education", "student", "teacher", "school", "learning", "study", "academic", "knowledge"],
    #     ["travel", "vacation", "tourist", "destination", "journey", "adventure", "culture", "explore"]
    # ]
    
    # documents = []
    # true_labels = []
    
    # docs_per_topic = n_docs // len(topics)
    
    # for topic_idx, words in enumerate(topics):
    #     for _ in range(docs_per_topic):
    #         # Generate document with 10-30 words
    #         doc_length = np.random.randint(10, 31)
    #         doc_words = np.random.choice(words, size=doc_length, replace=True)
            
    #         # Add some random common words
    #         common_words = ["the", "and", "is", "are", "for", "with", "this", "that"]
    #         noise_words = np.random.choice(common_words, size=np.random.randint(3, 8), replace=True)
            
    #         all_words = list(doc_words) + list(noise_words)
    #         np.random.shuffle(all_words)
            
    #         documents.append(" ".join(all_words))
    #         true_labels.append(topic_idx)
    
    # # Add remaining documents to reach exact count
    # remaining = n_docs - len(documents)
    # for _ in range(remaining):
    #     topic_idx = np.random.randint(len(topics))
    #     words = topics[topic_idx]
    #     doc_length = np.random.randint(10, 31)
    #     doc_words = np.random.choice(words, size=doc_length, replace=True)
    #     documents.append(" ".join(doc_words))
    #     true_labels.append(topic_idx)
    
    # # Add some empty documents to test handling
    # empty_count = n_docs // 100  # 1% empty documents
    # for _ in range(empty_count):
    #     documents.append("")
    #     true_labels.append(-1)  # Special label for empty docs
    
    # print(f"Generated {len(documents)} documents across {len(topics)} topics (with {empty_count} empty docs)")
    
    # Initialize clusterer with embeddings
    clusterer = AdvancedTextClusterer(
        memory_limit_gb=12, 
        batch_size=20000,  # Larger batches for efficiency
        embedding_model='sentence-transformers',  # Use best embeddings
        model_name='auto'  # Auto-select best model
    )
    
    print(f"\n🚀 Using embedding model: {clusterer.embedding_model}")
    if hasattr(clusterer, 'model_name'):
        print(f"📦 Model: {clusterer.model_name}")
        print(f"📐 Embedding dimension: {clusterer.embedding_dim}")
    
    # Perform clustering
    predicted_labels = clusterer.fit_predict(documents, n_clusters=None)
    
    if predicted_labels is not None:
        # Verify dimensions match
        print(f"\nDimension Check:")
        print(f"Input documents: {len(documents)}")
        print(f"Output labels: {len(predicted_labels)}")
        print(f"Dimensions match: {len(documents) == len(predicted_labels)}")
        
        # Only proceed if dimensions match
        if len(documents) == len(predicted_labels):
            # Filter out empty documents for accuracy calculation
            # valid_mask = (predicted_labels != -1) & (np.array(true_labels) != -1)
            # valid_true = np.array(true_labels)[valid_mask]
            # valid_pred = predicted_labels[valid_mask]
            
            # # Calculate accuracy
            # if len(valid_pred) > 0:
            #     from sklearn.metrics import adjusted_rand_score
            #     ari = adjusted_rand_score(valid_true, valid_pred)
            #     print(f"Adjusted Rand Index: {ari:.3f}")
            # else:
            #     print("No valid predictions to evaluate")
            
            # print(f"True clusters: {len(topics)}")
            print(f"Found clusters: {len(set(predicted_labels[predicted_labels != -1]))}")
            
            # Get comprehensive cluster descriptions
            descriptions = clusterer.generate_cluster_descriptions(documents)
            
            if descriptions:
                print("\n" + "="*80)
                print("COMPREHENSIVE CLUSTER DESCRIPTIONS")
                print("="*80)
                
                for cluster_id, info in descriptions.items():
                    print(f"\n📊 CLUSTER {cluster_id}")
                    print("-" * 50)
                    print(f"📝 Description: {info['description']}")
                    print(f"📈 Size: {info['size']} documents")
                    
                    if info['themes']:
                        print(f"🎯 Themes: {', '.join(info['themes'])}")
                    
                    if info['keywords']:
                        keywords_str = ', '.join([f"{kw[0]}" for kw in info['keywords'][:8]])
                        print(f"🔑 Key Terms: {keywords_str}")
                    
                    if info['phrases']:
                        phrases_str = ', '.join([f"'{phrase[0]}'" for phrase in info['phrases'][:3]])
                        print(f"💬 Common Phrases: {phrases_str}")
                    
                    print(f"📊 Avg Document Length: {info['statistics']['avg_document_length']:.1f} words")
                    
                    print("📄 Sample Documents:")
                    for i, sample in enumerate(info['sample_texts'][:2], 1):
                        preview = str(sample)[:100] + "..." if len(str(sample)) > 100 else str(sample)
                        print(f"   {i}. {preview}")
                    print()
            
            # Also show the simple keyword summary for comparison
            # summaries = clusterer.get_cluster_keywords(documents)
            
            # if summaries:
            #     print("\n" + "="*60)
            #     print("QUICK KEYWORD SUMMARY")
            #     print("="*60)
            #     for cluster_id, info in summaries.items():
            #         print(f"Cluster {cluster_id} ({info['size']} docs): {', '.join(info['keywords'][:5])}")
            
            # Lightweight visualization
            clusterer.plot_results_lightweight()
            
            return clusterer
        else:
            print("ERROR: Dimension mismatch between input and output!")
            return None
    else:
        print("Clustering failed!")
        return None

# Run the large-scale demonstration
# if __name__ == "__main__":
#     # Test with 25k documents using embeddings
#     clusterer = demo_large_scale_clustering(25000)


# COMMAND ----------

print(json_text)

# COMMAND ----------

sys_msg = '''
As a skills specialist, extract all demonstrable and practicable human-driven skills from input text.
These skills must reflect capabilities that can be observed, practiced, and refined.
Include the application of knowledge only when it results in a demonstrable and refinable skill.
Do not include standalone knowledge, abilities, or tasks. Just output the skills separated by "\n".
'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

parent_folder = Path("/Volumes/jsa_external_prod/external_vols/scratch/Scratch/Ehsan/NST")
input_folder = parent_folder / "Output"
output_folder = parent_folder / "Results"
output_folder.mkdir(exist_ok=True, parents=True)
meta_file = parent_folder / 'Meta/Sample_UOC.csv'
input_files = list(input_folder.glob("*.csv"))
all_df = []
for f in input_files:
    all_df.append(pd.read_csv(f))

# COMMAND ----------

df = pd.concat(all_df)
df.reset_index(drop=True, inplace=True)
display(df)

# COMMAND ----------

df_meta = pd.read_csv(meta_file).rename({'UOC': 'Unit Code'}, axis='columns')
df_meta = df_meta.drop_duplicates(subset=['Unit Code', 'QUAL', 'ANZSCO'])
df_ = df.merge(df_meta, on='Unit Code')
df_.reset_index(drop=True, inplace=True)
df_['Context'] = df_['Context'].apply(lambda x: x.split(':')[1] if ':' in x else x)
df_.columns, df_.shape

# COMMAND ----------

display(df_)

# COMMAND ----------

version = 'v8'
def get_sys_msg(qual, occp):
    sys_msg =  f'''
        Your goal is to extract the main **occupational skill** (NOT TASK) from the text below Must be "in a few words". The extracted skill MUST be aligned with the definition of the skill and you MUST put the "context" of Occupation and Education in the output SKILL. If you find ":" in the input text, MUST consider the input definition after ":" in the output SKILL:

        Definition of Skill: "Skill is a valued and purpose-driven human ability that is acquired or refined through learning and practice. It is a dynamic function of an individual’s knowledge, experience, and personal attributes that is continuously influenced by context, interaction with others, and the demands of the environment in which it is exercised. Also the skill MUST show the context or nuance of the **education** and **occupation**."

        Education: "{qual}"

        Occupations: "{occp}"

        Output format: combine <CONTEXT> and <SKILL> as a meaningful SKILL

        Note: 
            -Do not put the exact occupation or education or the word "contxt" or "setting" in the CONTEXT.
            -If the SKILL shows the context around the given Education or Occupation, then output <SKILL> only.
            -MUST Remove ":" in the output SKILL.
            -Do not generate any extra sentences.

        '''
    return sys_msg
responses = extarct_(df_, 'Context', llm, cols=['QUAL', 'ANZSCO TITLE'])
responses

# COMMAND ----------

version = 'v9'
def get_sys_msg(qual, occp):
    sys_msg =  f'''
        Your goal is to extract the main **occupational skill** (NOT TASK) from the text below Must be "in a few words". The extracted skill MUST be aligned with the definition of the skill and you MUST put the "context" of Occupation and Education in the output SKILL. If you find ":" in the input text, MUST consider the input definition after ":" in the output SKILL:

        Definition of Skill: "Skill is a valued and purpose-driven human ability that is acquired or refined through learning and practice. It is a dynamic function of an individual’s knowledge, experience, and personal attributes that is continuously influenced by context, interaction with others, and the demands of the environment in which it is exercised. Also the skill MUST show the context or nuance of the **education** and **occupation**."

        Education: "{qual}"

        Occupations: "{occp}"

        Output format: 

        The grammatical structure of skills that show the context around a job or education typically combines a core skill with contextual modifiers that clarify how, where, or in what setting the skill is applied.
        Here’s a breakdown of the common grammatical structure:
        ________________________________________
        General Structure:
        [Verb/Action] + [Object (if any)] + [Context/Qualifier]
        Or:
        [Noun-based Skill] + [Context/Qualifier]
        ________________________________________
        Examples with Explanation:
        1.	"Providing trauma-informed care in community settings"
        o	Verb: Providing
        o	Object: trauma-informed care
        o	Context: in community settings
        2.	"Developing lesson plans aligned with curriculum standards"
        o	Verb: Developing
        o	Object: lesson plans
        o	Context: aligned with curriculum standards
        3.	"Conflict resolution in multicultural workplaces"
        o	Skill (noun phrase): Conflict resolution
        o	Context: in multicultural workplaces
        4.	"Using data analytics to improve student performance"
        o	Verb: Using
        o	Object: data analytics
        o	Purpose/Context: to improve student performance
        5.	"Collaborative problem-solving in remote teams"
        o	Skill: Collaborative problem-solving
        o	Context: in remote teams
        ________________________________________
        Common Contextual Modifiers:
        •	Location/Setting: in aged care, in clinical environments, in rural schools
        •	Purpose: to support learning, to improve engagement
        •	Standard/Framework: aligned with industry standards
        •	Tools/Techniques Used: using XYZ software, through participatory methods
        •	Population: with refugees, with neurodiverse students


        Note: 
            -Do not put the exact occupation or education or the word "context" or "setting" in the CONTEXT.
            -If the SKILL shows the context around the given Education or Occupation, then output <SKILL> only.
            -MUST Remove ":" in the output SKILL.
            -Do not generate any extra sentences.

        '''
    return sys_msg
responses = extarct_(df_, 'Context', llm, cols=['QUAL', 'ANZSCO TITLE'])
responses

# COMMAND ----------

version = 'v10'
def get_sys_msg(qual, occp):
    sys_msg =  f'''
        Your goal is to extract the main **occupational skill** (NOT TASK) from the text below Must be "in a few words". The extracted skill MUST be aligned with the definition of the skill and you MUST put the "context" of Occupation and Education in the output SKILL. If you find ":" in the input text, MUST consider the input definition after ":" in the output SKILL:

        Definition of Skill: "Skill is a valued and purpose-driven human ability that is acquired or refined through learning and practice. It is a dynamic function of an individual’s knowledge, experience, and personal attributes that is continuously influenced by context, interaction with others, and the demands of the environment in which it is exercised. Also the skill MUST show the context or nuance of the **education** and **occupation**."

        Education: "{qual}"

        Occupations: "{occp}"

        Output format: 

        The grammatical structure of skills that show the context around a job or education typically combines a core skill with contextual modifiers that clarify how, where, or in what setting the skill is applied.
        Here’s a breakdown of the common grammatical structure:
        ________________________________________
        General Structure:
        [Verb/Action] + [Object (if any)] + [Context/Qualifier]
        Or:
        [Noun-based Skill] + [Context/Qualifier]
        ________________________________________
        Examples with Explanation:
        1.	"Providing trauma-informed care in community settings"
        o	Verb: Providing
        o	Object: trauma-informed care
        o	Context: in community settings
        2.	"Developing lesson plans aligned with curriculum standards"
        o	Verb: Developing
        o	Object: lesson plans
        o	Context: aligned with curriculum standards
        3.	"Conflict resolution in multicultural workplaces"
        o	Skill (noun phrase): Conflict resolution
        o	Context: in multicultural workplaces
        4.	"Using data analytics to improve student performance"
        o	Verb: Using
        o	Object: data analytics
        o	Purpose/Context: to improve student performance
        5.	"Collaborative problem-solving in remote teams"
        o	Skill: Collaborative problem-solving
        o	Context: in remote teams
        ________________________________________
        Common Contextual Modifiers:
        •	Location/Setting: in aged care, in clinical environments, in rural schools
        •	Purpose: to support learning, to improve engagement
        •	Standard/Framework: aligned with industry standards
        •	Tools/Techniques Used: using XYZ software, through participatory methods
        •	Population: with refugees, with neurodiverse students


        Note: 
            -Do not put the exact occupation or education or the word "context" or "setting" in the CONTEXT.
            -If the SKILL shows the context around the given Education or Occupation, then output <SKILL> only.
            -MUST Remove ":" in the output SKILL.
            -Do not generate any extra sentences.

        '''
    return sys_msg
responses = extarct_(df_, 'Context', llm, cols=['QUAL', 'ANZSCO TITLE'])
responses

# COMMAND ----------

sys_msg = f'''You are a helpful super intellignet AI assistant. Extract a single "Skill" from the input text as output. DO NOT generate any extra sentences.'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

sys_msg = '''
As a skills specialist, extract all demonstrable and practicable human-driven skills from input text.
These skills must reflect capabilities that can be observed, practiced, and refined.
Include the application of knowledge only when it results in a demonstrable and refinable skill.
Do not include standalone knowledge, abilities, or tasks. Just output the skills separated by "\n".
'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

version = 'v1'
sys_msg = '''
Your goal is to extract **occupational skills** from the text below. These skills should reflect applied capabilities aligned with **AQF Level 5 outcomes** — including judgement, problem-solving, and application of knowledge across varied contexts. Avoid listing tasks; focus on transferable, conceptual workplace skills.

Start by identifying the **first verb-object phrase** in each performance criterion. Use this as the anchor, but rewrite it into a **detailed, AQF-aligned skill** that reflects autonomy and responsibility.

Note: Output the skills separated by "\n". Do not generate any extra sentences.
'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

version = 'v2'
sys_msg = '''
Your goal is to extract the main **occupational skill** from the text below. These skill should reflect applied capabilities aligned with **AQF Level 5 outcomes** — including judgement, problem-solving, and application of knowledge across varied contexts. Avoid listing tasks; focus on transferable, conceptual workplace skill.

Start by identifying the **first verb-object phrase** in the text. Use this as the anchor, but rewrite it into a **detailed, AQF-aligned skill** that reflects autonomy and responsibility.

Note: Do not generate any extra sentences.
'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

version = 'v3'
sys_msg = '''
Your goal is to extract the main **occupational skill** from the text below in "a few words". These skill should reflect applied capabilities aligned with **AQF Level 5 outcomes** — including judgement, problem-solving, and application of knowledge across varied contexts. Avoid listing tasks; focus on transferable, conceptual workplace skill.

Start by identifying the **first verb-object phrase** in the text. Use this as the anchor, but rewrite it into a **detailed, AQF-aligned skill** that reflects autonomy and responsibility.

Note: Do not generate any extra sentences.
'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

version = 'v4'
sys_msg = '''
Your goal is to extract the main **occupational skill** from the text below in "a few words". These skill should reflect applied capabilities aligned with **AQF Level 5 outcomes** — including judgement, problem-solving, and application of knowledge across varied contexts. Avoid listing tasks; focus on transferable, conceptual workplace skill.

Start by identifying the **first verb-object phrase** in the text. Use this as the anchor, but rewrite it into a **detailed, AQF-aligned skill**.

Output format: action-verb + object + modifier (e.g. “Assist clients with daily living activities in a respectful and dignified manner”).

Note: Do not generate any extra sentences.
'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

version = 'v5'
sys_msg = '''
Your goal is to extract the main **occupational skill** from the text below in "a few words". These skill should reflect applied capabilities aligned with **AQF Level 5 outcomes** — including judgement, problem-solving, and application of knowledge across varied contexts. Avoid listing tasks; focus on transferable, conceptual workplace skill.

Start by identifying the **first verb-object phrase** in the text. Use this as the anchor, but rewrite it into a **detailed, AQF-aligned skill**.

Output format: action-verb + object + modifier (in a few words: e.g. “Assist clients with daily living activities in a respectful and dignified manner”).

Note: Do not generate any extra sentences.
'''
responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

version = 'v6'

sys_msg =  '''
Your goal is to extract the main **occupational skill** (NOT TASK) from the text below in "a few words". The extracted skill MUST be aligned with the definition of the skill as:

"Skill is a valued and purpose-driven human ability that is acquired or refined through learning and practice. It is a dynamic function of an individual’s knowledge, experience, and personal attributes that is continuously influenced by context, interaction with others, and the demands of the environment in which it is exercised. Also the skill MUST show the context or nuance of the **education** or **occupation**."

Education: Certificate III in Community Services, Health Support and Digital Technologies
Occupations: Community Support Worker or Nurse Assistant

Output format: noun + verb (in a few words: e.g. “Care Planning”, “Language Interpretation”).

Note: Do not generate any extra sentences.

'''

responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

version = 'v8'

sys_msg =  '''
Your goal is to extract the main **occupational skill** (NOT TASK) from the text below. The extracted skill MUST be aligned with the definition of the skill and you MUST put the "context" of Occupation and Education in the output SKILL:

Definition of Skill: "Skill is a valued and purpose-driven human ability that is acquired or refined through learning and practice. It is a dynamic function of an individual’s knowledge, experience, and personal attributes that is continuously influenced by context, interaction with others, and the demands of the environment in which it is exercised. Also the skill MUST show the context or nuance of the **education** and **occupation**."

Education: "Certificate III in Community Services, Health Support and Digital Technologies"

Occupations: "Community Support Worker or Nurse Assistant"

Skill format: noun + verb (in a few words: e.g. “Community Care Planning”, “Language Interpretation”).
Output format: <CONTEXT> <SKILL>

Note: 
    -Do not put the exact occupation or education or the word "contxt" or "setting" in the CONTEXT.
    -If the SKILL shows the context around the given Education or Occupation, then output <SKILL> only.
    -If you find ":" in the input text try to apply the definition after ":" in the output SKILL.
    -Do not generate any extra sentences.

'''

responses = extarct_(df_, 'Source text', llm, sys_msg)

# COMMAND ----------

responses

# COMMAND ----------

res = [[i] + [c.strip()] + [version] for i, s in enumerate(responses) for c in s.split('\n') if c.strip()!="" and ":" not in c.strip()]
df_res = pd.DataFrame(res, columns=['index', 'extracted_skill', 'prompt_version'])
df_res

# COMMAND ----------

cols = ['Unit name',
 'Skill name',
 'Source text',
 'Source section',
 'UUID']
display(df_.reset_index().merge(df_res, on='index'))
df_.columns.tolist()

# COMMAND ----------


import time
while True:
    time.sleep(240)

# COMMAND ----------

# MAGIC %md
# MAGIC # TEST

# COMMAND ----------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse a UC unit HTML page and output only the course object:

{
  "code": "…",
  "name": "…",
  "description": "…",
  "study_level": "Level 3 - Undergraduate Advanced Unit",
  "learning_outcomes": [...],
  "prerequisites": [...],
  "credit_points": <float or null>,
  "topics": [...],
  "assessment": <string or null>
}

Notes
- No external libraries required.
- 'study_level' is taken from the table's 'Study level' cell.
- Credit points are read from the 'Credit points' cell (prefers integers like 3/6/12 over EFTSL decimals).
"""

import os
import re
import json
from html import unescape
from typing import List, Optional

# ---------- Config ----------
HTML_PATH = "/mnt/data/Digital Marketing (11179) - University of Canberra.html"  # change to your HTML path

# ---------- HTML -> text ----------
def html_to_text(html: str) -> str:
    html = re.sub(r'<script\b[^>]*>[\s\S]*?</script>', ' ', html, flags=re.I)
    html = re.sub(r'<style\b[^>]*>[\s\S]*?</style>', ' ', html, flags=re.I)
    # preserve structure with newlines
    html = re.sub(r'</(p|div|li|tr|th|td|h[1-6])\s*>', '\n', html, flags=re.I)
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
    # mark list items before stripping tags
    html = re.sub(r'<li\b[^>]*>', '\n- ', html, flags=re.I)
    # strip tags
    html = re.sub(r'<[^>]+>', ' ', html)
    # unescape + whitespace normalize
    text = unescape(html).replace('\r', '\n')
    text = re.sub(r'\u00a0', ' ', text)          # non-breaking space
    text = re.sub(r'[\t ]+', ' ', text)          # collapse spaces
    text = re.sub(r'\n{2,}', '\n\n', text)       # collapse blank lines
    return text.strip()

def get_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r'^[•·▪◦*]+', '- ', ln)          # normalize bullets
        ln = re.sub(r'^\(?\d+\)?[.)]\s+', '- ', ln)  # numbered -> "- "
        cleaned.append(ln)
    return cleaned

def find_first(pattern: str, text: str, flags=re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

# ---------- Section helper ----------
def extract_section(text: str, start_patterns, stop_patterns) -> Optional[str]:
    lines = get_lines(text)
    start_idx = None
    for i, ln in enumerate(lines):
        for pat in start_patterns:
            if re.search(pat, ln, flags=re.I):
                start_idx = i
                break
        if start_idx is not None:
            break
    if start_idx is None:
        return None
    out = []
    for j in range(start_idx + 1, len(lines)):
        ln = lines[j]
        if any(re.search(p, ln, flags=re.I) for p in stop_patterns):
            break
        out.append(ln)
    return "\n".join(out).strip() if out else None

# ---------- Field parsers ----------
def parse_course_code(text: str, filename: str = "") -> Optional[str]:
    for pat in [
        r'\b(?:Unit|Course)\s*Code\s*[:\-]?\s*([A-Z]{3,4}\s?\d{3,4}|\d{5}(?:\.\d)?)',
        r'\bCode\s*[:\-]?\s*([A-Z]{3,4}\s?\d{3,4}|\d{5}(?:\.\d)?)',
        r'\((\d{5}(?:\.\d)?)\)'  # e.g., "Digital Marketing (11179.3)"
    ]:
        m = find_first(pat, text)
        if m:
            return m.replace(" ", "")
    if filename:
        m = re.search(r'(\d{5}(?:\.\d)?)', filename)
        if m:
            return m.group(1)
    return None

def parse_course_name(text: str) -> Optional[str]:
    # "<Name> (11179.3)"
    m = re.search(r'\n?\s*([A-Z][A-Za-z \-&\'"\u2019]+?)\s*\(\s*\d{5}(?:\.\d)?\s*\)', text)
    if m:
        name = m.group(1).strip()
        if len(name.split()) >= 2 and "University of Canberra" not in name:
            return name
    # fallback: plausible heading near top
    lines = get_lines(text)
    candidates = []
    for ln in lines[:60]:
        low = ln.lower()
        if not ln or any(w in low for w in ['credit points', 'assessment', 'learning outcome', 'introduction', 'prereq']):
            continue
        if len(ln.split()) >= 2 and len(ln) <= 120:
            candidates.append(ln)
    return sorted(candidates, key=len, reverse=True)[0] if candidates else None

def parse_description(text: str) -> Optional[str]:
    section = extract_section(
        text,
        start_patterns=[r'^\s*Introduction\s*$', r'Unit\s+Description', r'^\s*Description\s*$'],
        stop_patterns=[r'learning outcomes?', r'graduate attributes', r'prerequisites?']
    )
    if section:
        para = re.sub(r'\s*\n\s*', ' ', section)
        return re.sub(r'\s{2,}', ' ', para).strip()
    return None

def parse_topics_from_intro(description: Optional[str]):
    if not description:
        return []
    m = re.search(r'Covered topics include\s+(.+?)[\.\!]\s', description, flags=re.I)
    if not m:
        m = re.search(r'include\s+(.+?)[\.\!]$', description, flags=re.I)
    if not m:
        return []
    s = m.group(1)
    parts = re.split(r',|\band\b', s)
    return [p.strip(' .;') for p in parts if p.strip()]

def parse_learning_outcomes(text: str):
    section = extract_section(
        text,
        start_patterns=[r'learning outcomes?'],
        stop_patterns=[r'graduate attributes', r'prerequisites?', r'availability', r'assessment']
    )
    if not section:
        return []
    s = re.sub(r'\s*(\d+)\.\s+', r'\n\1. ', section).strip()
    items = []
    for line in s.splitlines():
        m = re.match(r'\d+\.\s*(.+)', line)
        if m:
            items.append(m.group(1).strip(' .;'))
        elif line.strip().startswith('- '):
            items.append(line.strip()[2:].strip(' .;'))
    # de-duplicate
    out, seen = [], set()
    for it in items:
        t = re.sub(r'\s{2,}', ' ', it)
        if t and t.lower() not in seen:
            out.append(t)
            seen.add(t.lower())
    return out

def parse_prerequisites(text: str):
    section = extract_section(
        text,
        start_patterns=[r'^\s*Prerequisites\s*$'],
        stop_patterns=[r'^\s*Corequisites\s*$', r'^\s*Incompatible units\s*$', r'^\s*Equivalent units\s*$', r'^\s*Assumed knowledge\s*$']
    )
    if not section:
        return []
    parts = re.split(r',|;|\band\b|\n', section, flags=re.I)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() not in ['none', 'nil']]

def parse_credit_points(text: str) -> Optional[float]:
    """Prefer the integer immediately after the 'Credit points' label; avoid EFTSL decimals (0.125)."""
    lines = get_lines(text)
    for i, ln in enumerate(lines):
        if re.fullmatch(r'(?i)credit points', ln):
            candidates = []
            for j in range(i+1, min(i+40, len(lines))):
                for m in re.finditer(r'\b([0-9]{1,2}(?:\.[0-9]+)?)\b', lines[j]):
                    token = m.group(1)
                    val = float(token)
                    candidates.append((j, token, val, token.isdigit()))
            ints = [c for c in candidates if c[3]]
            if ints:
                ints.sort(key=lambda x: x[0])  # earliest int after label
                return float(ints[0][2])
            if candidates:
                candidates.sort(key=lambda x: x[0])
                return float(candidates[0][2])
    m = re.search(r'(?is)credit\s*points[\s:.-]*([0-9]{1,2}(?:\.[0-9]+)?)', text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None

def parse_study_level(text: str) -> Optional[str]:
    """Extract 'Study level' cell, favoring lines that look like a level descriptor."""
    lines = get_lines(text)
    for i, ln in enumerate(lines):
        if re.fullmatch(r'(?i)study level', ln):
            # Typical UC value examples: "Level 3 - Undergraduate Advanced Unit"
            for j in range(i+1, min(i+50, len(lines))):
                cand = lines[j].strip()
                if not cand:
                    continue
                if re.search(r'(?i)^(level\\s*\\d|undergraduate|postgraduate|honours|research)', cand):
                    return cand
            # fallback: first non-empty after the header trio row
            non_empty = [lines[j].strip() for j in range(i+1, min(i+50, len(lines))) if lines[j].strip()]
            if non_empty:
                return non_empty[0]
    m = re.search(r'(?i)study\\s*level\\s*[:\\-]?\\s*(.+)', text)
    if m:
        return m.group(1).strip()
    return None

def parse_assessment(text: str) -> Optional[str]:
    lines = get_lines(text)
    items = [ln for ln in lines if re.search(r'\b\d{1,3}\s?%\b', ln)]
    if not items:
        return None
    items = [re.sub(r'\s{2,}', ' ', it).strip() for it in items]
    return '; '.join(items)

def extract_course_object_from_html(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()
    text = html_to_text(html)
    filename = os.path.basename(path)
    description = parse_description(text)

    return {
        "code": parse_course_code(text, filename),
        "name": parse_course_name(text),
        "description": description,
        "study_level": parse_study_level(text),
        "learning_outcomes": parse_learning_outcomes(text),
        "prerequisites": parse_prerequisites(text),
        "credit_points": parse_credit_points(text),
        "topics": parse_topics_from_intro(description),
        "assessment": parse_assessment(text)
    }

if __name__ == "__main__":
    obj = extract_course_object_from_html(HTML_PATH)
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# COMMAND ----------

{
  "code": "11179",
  "name": "Digital Marketing",
  "description": "This unit serves as a bridge between new technologies and relevant areas of existing knowledge. It develops a framework for understanding the forces propelling the digital revolution in marketing and business. The unit examines the contemporary digital marketing landscape by understanding and applying fundamental concepts of digital marketing. Covered topics include the fundamentals of effective digital marketing strategy, online consumer behaviour, online marketing communication, content marketing, artificial intelligence and digital analytics. Through a blend of theoretical concepts and practical applications, participants will learn how to optimise online presence and analyses key metrics for data-driven decision-making. Upon completion of this unit, participants will be proficient in creating compelling online content, targeting specific audiences, and adapting strategies to leverage emerging trends in the digital landscape.",
  "year": 2025,
  "learning_outcomes": [
    "Understand and apply the fundamental concepts used in digital marketing",
    "Analyse digital marketing and AI tools to assess the effectiveness of marketing campaigns and make data-driven decisions",
    "Evaluate the ethical, legal and societal implications of digital marketing practices on the future of workforce; and",
    "Utilise digital marketing tools and techniques to develop effective marketing campaigns"
  ],
  "prerequisites": [
    "11176 Marketing Fundamentals"
  ],
  "credit_points": 3.0,
  "topics": [
    "the fundamentals of effective digital marketing strategy",
    "online consumer behaviour",
    "online marketing communication",
    "content marketing",
    "artificial intelligence",
    "digital analytics"
  ],
  "assessment": null
}
