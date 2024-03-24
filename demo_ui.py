import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import requests
import base64
import hashlib
import torch
import time
import re
import tempfile
import shutil
import pandas as pd
from langchain_community.chat_models import BedrockChat
from func_v2 import agent_executor
from func_v2 import bedrock_llm
from func_v2 import retriever
from func_v2 import retrievalQA
from func_v2 import chatModel
from func_v2 import boto3_bedrock
from func_v2 import run_vqa_prompt

df = pd.read_json('./role_template.json',orient='records')
DESCRIPTION = '''<h2 style='text-align: center'> 企业搜索问答demo </h2>'''
default_chatbox = [("", "有什么可以帮您?")]
role_keys = []
role_values = []
role_prompt_dict={}
tmpdir="./"
cur_role=""
model_id = None  # 新增全局变量用于存储模型 ID
models_map = {"claude3-haidu":"anthropic.claude-3-haiku-20240307-v1:0",
              "claude3-sonnet":"anthropic.claude-3-sonnet-20240229-v1:0"}


def delete_files_in_directory(directory):
    file_names = os.listdir(directory)
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def update_df(table_data):
    # 更新DataFrame
    global df
    df = table_data
    return table_data

def save_df(table_data):
    global df
    # 将DataFrame保存为json文件
    df.to_json('./role_template.json',orient='records',force_ascii=False)
    return df


#####authentication#########
def auth_fn(user, password):
    return user == 'admin' and password == 'admin123'



#####initial role prompt template#####
def initial_role_prompt(role_template_path:str):
    global role_keys,role_values,role_prompt_dict
    with open(role_template_path) as f:
        json_data = json.load(f)
        for item in json_data:
            role_keys.append(item["role"])
            role_values.append(item["instruct"])
            role_prompt_dict[item["role"]]=item["instruct"]




######gradio component func############
def update_textbox(value):
    global cur_role
    cur_role = value
    return role_prompt_dict[value]

def execute_agent(query:str,instruct:str,chat_history):
    global cur_role
    prompt = instruct + "\n" +query
    bot_msg = ""
    #bot_msg =  agent_executor.run(prompt)
    files = os.listdir(tmpdir)
    if  "图文问答" in cur_role :
        input_image = retriever._get_image_file(tmpdir)
        bot_msg = run_vqa_prompt(boto3_bedrock, input_image, prompt, 1000)
    elif len(files)>0:
        retriever._get_content_type(tmpdir)
        print("doc found!")
        bot_msg = retrievalQA.run(prompt)
    else:
        #bot_msg = bedrock_llm.predict(prompt)
        bot_msg = chatModel(prompt)
    response = (query, bot_msg)
    chat_history.append(response)
    return "",chat_history


def generate_file(file_obj):
    global tmpdir
    print('临时文件夹地址：{}'.format(tmpdir))
    print('上传文件的地址：{}'.format(file_obj.name)) # 输出上传后的文件在gradio中保存的绝对地址

    #获取到上传后的文件的绝对路径后，将文件复制到临时目录中
    shutil.copy(file_obj.name, tmpdir)

    # 获取上传Gradio的文件名称
    FileName=os.path.basename(file_obj.name)

    # 获取拷贝在临时目录的新的文件地址
    NewfilePath=os.path.join(tmpdir,FileName)
    retriever._get_content_type(tmpdir)
    return NewfilePath

    # 打开复制到新路径后的文件
    #with open(NewfilePath, 'rb') as file_obj:
    #    #在本地打开一个新的文件，并且将上传文件内容写入到新文件
    #    outputPath=os.path.join("./docs/",FileName)
    #    with open(outputPath,'wb') as w:
    #        w.write(file_obj.read())
    #return outputPath




def clear_fn(value):
    delete_files_in_directory(tmpdir)
    return "", default_chatbox,None

def save_model_id(model_dropdown):
    global model_id
    model_id = models_map[model_dropdown]
    print(f"Selected model ID: {model_id}")
    chatModel = re_initial(model_id)
    retrievalQA = RetrievalQA.from_llm(llm=chatModel, retriever=retriever)



def main():
    gr.close_all()
    global tmpdir,role_keys
    with tempfile.TemporaryDirectory(dir='./tmp/') as tmpdir:
        with gr.Blocks(css='style.css') as demo:
            with gr.Tab("main"):
                gr.Markdown(DESCRIPTION)
                with gr.Row():
                    with gr.Column(scale=4.5):
                        with gr.Group():
                            input_text = gr.Textbox(label='你想问点什么？', placeholder='Please enter text prompt below and press ENTER.')
                            with gr.Row():
                                run_button = gr.Button('提交')
                                clear_button = gr.Button('清除')
                            with gr.Row():
                                doc_inputs = gr.components.File(label="上传文件")


                    with gr.Column(scale=5.5):
                        result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "有什么可以帮您?")],height=550)
                with gr.Row():
                    dropdown = gr.Dropdown(role_prompt_dict,value="日语翻译")
                    instuct_text = gr.Textbox(role_values[0],visible=False)
            with gr.Tab("role"):
                data_table = gr.Dataframe(value=df, interactive=True)
                save_button = gr.Button("Save")
            with gr.Tab("model"):  # 新增一个 Tab
                model_dropdown = gr.Dropdown(
                    choices=["claude3-haidu", "claude3-sonnet"],
                    value="claude3-haidu",
                    label="Select Model"
                )
                save_model_button = gr.Button("Save Model")

            ###控件事件handler#####
            save_button.click(fn=save_df, inputs=[data_table], outputs=[data_table])
            data_table.change(fn=update_df, inputs=[data_table], outputs=None)
            dropdown.change(fn=update_textbox, inputs=dropdown, outputs=instuct_text)
            run_button.click(fn=execute_agent,inputs=[input_text,instuct_text,result_text],outputs=[input_text,result_text])
            input_text.submit(fn=execute_agent,inputs=[input_text,instuct_text,result_text],outputs=[input_text,result_text])
            clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text,doc_inputs])
            doc_inputs.upload(fn=generate_file,inputs=[doc_inputs], outputs=[doc_inputs])
            save_model_button.click(fn=save_model_id, inputs=model_dropdown, outputs=None)
            print(gr.__version__)

        #demo.queue(concurrency_count=10)
        demo.launch(share=True,auth=auth_fn,server_name='0.0.0.0')
        #demo.launch(share=True,server_name='0.0.0.0')
        #demo.launch(share=True)


if __name__ == '__main__':
    initial_role_prompt("./role_template.json")
    main()