# Databricks notebook source
# MAGIC %pip install -U vllm --force

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, login
import os
import ssl

# COMMAND ----------

# download the model form HF just once
if False:
    login(token='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', add_to_git_credential=True)
    model_id = "ISTA-DASLab/Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16"
    # model_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"
    # revision = "cf47bb3e18fe41a5351bc36eef76e9c900847c89"
    snapshot_location = snapshot_download(repo_id=model_id)
# %sh cp -L -r '/root/.cache/huggingface/hub/models--neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w4a16' '../../dbfs/mnt/scratch/Scratch/Ehsan/Models/'

# COMMAND ----------

# MAGIC %sh ls '/root/.cache/huggingface/hub/models--NousResearch--Hermes-3-Llama-3.1-70B-FP8/snapshots/6204699c12554bd184d2789c199284d37ab34194'

# COMMAND ----------

# MAGIC %sh cp -L -r '/root/.cache/huggingface/hub/models--ISTA-DASLab--Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16' '../../dbfs/mnt/scratch/Scratch/Ehsan/Models/'

# COMMAND ----------

# MAGIC %sh mkdir ../../local_disk0/model
# MAGIC cp -r /dbfs/mnt/scratch/Scratch/Ehsan/Models/models--neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w4a16 ../../local_disk0/model
# MAGIC ls ../../local_disk0/model

# COMMAND ----------

# MAGIC %sh mkdir ../../local_disk0/model
# MAGIC cp -r /dbfs/mnt/scratch/Scratch/Ehsan/Models/models--NousResearch--Hermes-3-Llama-3.1-70B-FP8 ../../local_disk0/model
# MAGIC ls ../../local_disk0/model

# COMMAND ----------

model_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"
revision = "8c670bcdb23f58a977e1440354beb7c3e455961d"
snapshot_location = snapshot_download(repo_id=model_id, revision=revision, cache_dir="../../local_disk0/model/.")

# COMMAND ----------

model_id = "NousResearch/Hermes-3-Llama-3.1-70B-FP8"
revision = "6204699c12554bd184d2789c199284d37ab34194"
snapshot_location = snapshot_download(repo_id=model_id, revision=revision, cache_dir="../../local_disk0/model/.")

# COMMAND ----------

snapshot_location = '/root/.cache/huggingface/hub/models--ISTA-DASLab--Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16/snapshots/4cea3ccca0a34fee1a20f9c94eada9d70f12c0b5'

# COMMAND ----------


number_gpus = 1
max_model_len = 4096#8192

# tokenizer = AutoTokenizer.from_pretrained(snapshot_location)
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"}
# ]
# prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
llm = LLM(model=snapshot_location, tensor_parallel_size=number_gpus, max_model_len=max_model_len)
# outputs = llm.generate(prompts, sampling_params)
# generated_text = outputs[0].outputs[0].text
# generated_text

# COMMAND ----------

sys_msg = '''You are a helpful super intellignet AI assistant.
            Extract ALL the information in the input text including the LIST of KEY information like 'FIRST NAME', 'MIDDLE NAME', 'LAST NAME', 'STREET ADDRESS', 'ZIP CODE', 'STATE', 'EMAIL', 'PHONE NUMBER' and 'CITY' in JSON format.

            Note:
            -Output in a JSON format only.
            -DO NOT generate any heading or trailing characters, just return the JSON format.
            -If there is no information for the KEY, then return an empty LIST for that KEY.
            -DO NOT generate any characters rather than the JSON output. No characters before or after curly brackets.
            -Make sure the key and values of the JSON output are enclosed in double quote.
            -DO NOT MISS the "start" and "end" curly brackets in the JSON format.
            -Make sure to find all the infromation. DOUBLE check the input.
            
            '''
# sys_msg = '''You are a helpful AI assistant, you must speak in JSON format only.
#             Extract information in the input text including KEY information like 'FIRST NAME', 'MIDDLE NAME', 'LAST NAME', 'STREET ADDRESS', 'ZIP CODE', 'STATE', 'EMAIL', 'PHONE NUMBER' and 'CITY' in JSON format.

#             For example, consider the following input text:

#             ``` Please call Ehsan Mohebi on 469 426 854 for any enquiries or drop an email at eh.mohebi@gmail.com. Ehsan is living in ACT 2914, Australia.
#             ```
#             Then the output must be in JSON format like below:
#             ```json{
#                 "FIRSTNAME": "Ehsan", 
#                 "MIDDLENAME": "",
#                 "LASTNAME": "Mohebi", 
#                 "STREETADDRESS": "", 
#                 "ZIPCODE": "2914", 
#                 "STATE": "ACT", 
#                 "EMAIL": "eh.mohebi@gmail.com", 
#                 "PHONE NUMBER": "469 426 854", 
#                 "CITY": ""
#             }```

#             Note:
#             -Output in a JSON format only.
#             -DO NOT generate any heading or trailing characters, just return the JSON format.
#             -If there is no information for the KEY, then return "" for that KEY.
#             -DO NOT generate any characters rather than the JSON output. No characters before or after curly brackets.
#             -Make sure the key and values of the JSON output are enclosed in double quote.
#             -DO NOT MISS the "start" and "end" curly brackets in the JSON format.
            
#             '''
# def instruction_format(text):
#     return f'''You are a helpful AI assistant, you must speak in JSON format only.
#             Read the following text from job description:
#             ```
#             {text}'''+'''
#             ```
#             Extract information in the above text including KEY information: 'FIRST NAME', 'MIDDLE NAME', 'LAST NAME', 'STREET ADDRESS', 'ZIP CODE', 'STATE', 'EMAIL', 'PHONE NUMBER' and 'CITY' in JSON format.

#             For example, consider the following input text:

#             ``` Please call Ehsan Mohebi on 469 426 854 for any enquiries or drop an email at eh.mohebi@gmail.com. Ehsan is living in ACT 2914, Australia.
#             ```
#             Then the output must be in JSON format like below:
#             ```json{
#                 "FIRSTNAME": "Ehsan", 
#                 "MIDDLENAME": "",
#                 "LASTNAME": "Mohebi", 
#                 "STREETADDRESS": "", 
#                 "ZIPCODE": "2914", 
#                 "STATE": "ACT", 
#                 "EMAIL": "eh.mohebi@gmail.com", 
#                 "PHONE NUMBER": "469 426 854", 
#                 "CITY": ""
#             }```

#             Note:
#             -Output in a JSON format only.
#             -DO NOT generate any heading or trailing characters, just return the JSON format.
#             -If there is no information for the KEY, then return "" for that KEY.
#             -DO NOT generate any characters rather than a JSON output. No characters before or after curly brackets.
#             -Make sure the key and values of the JSON output are enclosed in double quote.
#             -DO NOT MISS the "start" and "end" curly brackets in the JSON format.
            
#             '''

def instruction_format(sys_message: str, query: str):
    # note, don't "</s>" to the end
    return f'<s> [INST] {sys_message} [/INST]\nUser: {query}\nAssistant: '

def instruction_format(sys_message: str, query: str):
    # note, don't "</s>" to the end
    return f'<|im_start|>system\n{sys_message}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant'

# COMMAND ----------

t = """
Time Remaining 6 daysSouth Regional TAFELecturer Carpentry JoineryGrade 1 8 85 555 108 501 pa pro rata LGAPosition No 50001101Work Type Fixed Term Part TimeLocation BunburyClosing Date 2023 03 30 4 00 PM YYYY MM DD Up to 24 month full time part time 0 5 1 0 appointment with possible further fixed term and or permanent appointment South Regional TAFE delivers a wide range of nationally recognised programs throughout the Great Southern and South West regions of Western Australia The college delivers courses on site online and in the workplace and operates 12 campuses from Bunbury to Esperance and from Albany to Narrogin This significant role enables the college to respond to community and industry expectations in respect to the provision of vocational education and training qualifications Employees benefit from a broad range of professional development opportunities We are committed to being an equitable and diverse employer and encourages applications from Aboriginal and Torres Strait Islander peoples people from culturally diverse backgrounds young people and people with disability For more information visit www southregionaltafe wa edu auPlease note that applicants must be available for interview during the week commencing Monday 3 April 2023About the roleThe lecturer s primary role is to teach facilitate learning assess and mentor students in accordance with relevant curriculum and or training package requirements Lecturers also undertake Professional Activities and Activities Related to Delivery In order to maintain quality educational services lecturers are required to keep abreast of technological and other developments in their field through professional development to provide up to date information and advice to the College and industry where appropriate Relevant Qualification and ExperienceTo be eligible for employment applicants must hold aTrade Certificate III in Carpentry or equivalent and have a minimum of 5 years working in the area of expertise It is preferred that candidates already possess the minimum credentials required for trainers and assessors to deliver accredited vocational education and training VET as stipulated at 1 14 of the Standards for Registered Training Organisations RTOs 2015 or can demonstrate progression towards a relevant qualification South Regional TAFE prefers lecturers to hold the Certificate IV in Training and Assessment TAE40116 or TAE40110 and the additional units of competency in Schedule 1 Item 2 of the Standards for Registered Training Organisations RTOs 2015Applicants who do not yet hold the minimum required VET credentials are still eligible to apply however the recommended candidate may be required to obtain Certificate IV in Training and Assessment TAE40116 prior to appointment irrespective of holding other adult education qualifications For Further Job Related InformationPlease contact Alison O Neil Training Manager on 0417 004 583 or via email Alison O Neil srtafe wa edu auWork BenefitsOur employees have access to a range of benefits including Salary packaging sacrifice facility10 5 employer super contribution Thirteen weeks paid long service leave after 7 yearsFifteen days personal leaveFour weeks annual leaveFour weeks professional leaveFlexible working hoursPaid parental leave 14 weeksHow to Apply Please refer to the Job Description Form for full position details and the Application Information as these documents will assist you with the preparation of your application Applicants are required to apply online and submit by the closing date detailed below A comprehensive Resume including two professional refereesA written application addressing the selection criteria outlined in the attached Job Description Form Advertised Reference Number SRL033 23STo submit your application please click on the Apply for Job button If you are experiencing any technical difficulties please contact Establishment and Recruitment on 08 9203 3735 Other Conditions and EligibilityEligibility for employment is subject to obtaining a satisfactory Department of Education Criminal Clearance If this position involves contact with children the recommended occupant will also be required to obtain a Working with Children WWC Card To be eligible for appointment applicants must have a working visa for fixed term contract appointments or permanent residency for permanent appointments This selection process may be used to identify suitable applicants for similar vacancies that arise within the next twelve months Should no suitable applicant be identified or an offer declined the panel may search for further applicants beyond the closing date SR TAFE has a shutdown period of up to 12 working days over Christmas New Year Arrangements for paid leave in advance or leave without pay during the shutdown period will be negotiated with the successful applicant Please noteThe onus is on the applicant to ensure that their application is received by the closing date and time Late applications will not be accepted Applications close Thursday 30 March 2023 at 4 00pm WST ATTACHMENTS Application Information pdf lecturer jdf pdfYou can view and print these PDF attachments by downloading Adobe Reader 
"""

# COMMAND ----------


import pandas as pd
sampling_params = SamplingParams(temperature=0.6, top_p=1, max_tokens=1024)

file_path = '/dbfs/mnt/scratch/Scratch/Ehsan/temp1/SkillMatch_v2.csv'
df = pd.read_csv(file_path)
list_ = df["JobText"].values.tolist()

full_prompts = [instruction_format(sys_msg, t) for t in list_]
outputs = llm.generate(full_prompts, sampling_params=sampling_params)
responses = [output.text for out in outputs for output in out.outputs]
responses

# COMMAND ----------

import json
import re
all_text = []
error_collection = []
for i in range(len(responses)):
    t = list_[i]
    try:
        res = responses[i].replace("```", '').replace("\n", '')
        res = re.findall('{.*}', res)[0]
        map_ = json.loads(res) 
        for key in map_:
            if map_[key] == []: continue
            for item in map_[key]:
                t = t.replace(item, "<"+key+">")
    except:
        error_collection.append(responses[i])
    all_text.append([t, map_])
error_collection

# COMMAND ----------

all_text

# COMMAND ----------

display(df.join(pd.DataFrame(all_text)).rename({0:"AdText", 1:"RES"}, axis='columns').drop('JobText', axis='columns'))

# COMMAND ----------


