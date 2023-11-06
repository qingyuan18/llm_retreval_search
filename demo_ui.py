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
from func import agent_executor

DESCRIPTION = '''<h2 style='text-align: center'> 企业搜索问答demo </h2>'''
default_chatbox = [("", "有什么可以帮您?")]
role_keys = []
role_values = []
role_prompt_dict={}
tmpdir="./"


def find_index(array, value):
  for i, v in enumerate(array):
    if v == value:
      return i
  return -1

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
    global role_prompt_dict
    print("value=="+value)
    return value

def execute_agent(query:str,instruct:str):
    prompt = instruct + "\n" +query
    reply =  agent_executor.run(prompt)
    response = (reply, False)
    return [response]


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
    print(NewfilePath)

    # 打开复制到新路径后的文件
    with open(NewfilePath, 'rb') as file_obj:

        #在本地打开一个新的文件，并且将上传文件内容写入到新文件
        outputPath=os.path.join("./docs/",FileName)
        with open(outputPath,'wb') as w:
            w.write(file_obj.read())

    # 返回新文件的的地址（注意这里）
    return outputPath


def clear_fn(value):
    return "", default_chatbox


def main():
    gr.close_all()
    global tmpdir,role_keys
    with tempfile.TemporaryDirectory(dir='./tmp/') as tmpdir:
        with gr.Blocks(css='style.css') as demo:   
            gr.Markdown(DESCRIPTION)    
            with gr.Row():
                with gr.Column(scale=4.5):
                    with gr.Group():
                        input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
                        with gr.Row():
                            run_button = gr.Button('提交')
                            clear_button = gr.Button('清除')
                        with gr.Row():
                            doc_inputs = gr.components.File(label="上传文件")
                        with gr.Row():
                            doc_outputs = gr.components.File(label="下载文件",interactive=False,visible=False)
    
    
                with gr.Column(scale=5.5):
                    result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "有什么可以帮您?")],height=550)
            with gr.Row():
                dropdown = gr.Dropdown(role_prompt_dict)
                instuct_text = gr.Textbox(role_values[0],visible=False)
    
            ###控件事件handler#####
            dropdown.change(fn=update_textbox, inputs=dropdown, outputs=instuct_text) 
            run_button.click(fn=execute_agent,inputs=[input_text,instuct_text],outputs=[result_text])
            input_text.submit(fn=execute_agent,inputs=[input_text],outputs=[result_text])
            clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text])
            doc_inputs.upload(fn=generate_file,inputs=[doc_inputs], outputs=[doc_outputs])
            
            print(gr.__version__)
    
        #demo.queue(concurrency_count=10)
        demo.launch(share=True)


if __name__ == '__main__':
    initial_role_prompt("./role_template.json")
    main()