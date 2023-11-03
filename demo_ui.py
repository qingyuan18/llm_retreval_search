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


DESCRIPTION = '''<h2 style='text-align: center'> 企业搜索问答demo </h2>'''
default_chatbox = [("", "Hi, What can I do for you?")]
role_keys = []
role_values = []
with open('./role_template.json') as f:
    global role_keys,role_values
    json_data = json.load(f)
    role_keys = list(json_data.keys())
    role_values = list(json_data.values())




######gradio component func############
def update_textbox(key):
    return role_values[key]
    
from func import agent_executor
def execute_agent(query:str,instruct:str):
    prompt = instruct + "\n" +query
    return agent_executor.run(prompt)


def generate_file(file_obj):
    global tmpdir
    print('临时文件夹地址：{}'.format(tmpdir))
    print('上传文件的地址：{}'.format(file_obj.name)) # 输出上传后的文件在gradio中保存的绝对地址

    #获取到上传后的文件的绝对路径后，其余的操作就和平常一致了
    # 将文件复制到临时目录中
    shutil.copy(file_obj.name, tmpdir)

    # 获取上传Gradio的文件名称
    FileName=os.path.basename(file_obj.name)

    # 获取拷贝在临时目录的新的文件地址
    NewfilePath=os.path.join(tmpdir,FileName)
    print(NewfilePath)

    # 打开复制到新路径后的文件
    with open(NewfilePath, 'rb') as file_obj:

        #在本地打开一个新的文件，并且将上传文件内容写入到新文件
        outputPath=os.path.join(tmpdir,"New"+FileName)
        with open(outputPath,'wb') as w:
            w.write(file_obj.read())

    # 返回新文件的的地址（注意这里）
    return outputPath


def clear_fn(value):
    return "", default_chatbox


def main():
    gr.close_all()
    global tmpdir,role_keys
    with tempfile.TemporaryDirectory(dir='./docs/') as tmpdir:
        with gr.Blocks(css='style.css') as demo:   
            gr.Markdown(DESCRIPTION)    
            with gr.Row():
                with gr.Column(scale=4.5):
                    with gr.Group():
                        input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
                        with gr.Row():
                            run_button = gr.Button('Generate')
                            clear_button = gr.Button('Clear')
                        with gr.Row():
                            doc_inputs = gr.components.File(label="上传文件")
                        with gr.Row():
                            doc_outputs = gr.components.File(label="下载文件",interactive=False)
    
    
                with gr.Column(scale=5.5):
                    result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, what can I do for you?")]).style(height=550)
            with gr.Row():
                dropdown = gr.Dropdown(role_keys, value=role_keys[0], type="index")
                instuct_text = gr.Textbox(json_data[keys[0]],visible=False)
    
            ###控件事件handler#####
            dropdown.change(fn=update_textbox, inputs=dropdown, outputs=instuct_text) 
            run_button.click(fn=execute_agent,inputs=[input_text,instuct_text],outputs=[input_text, result_text])
            input_text.submit(fn=execute_agent,inputs=[input_text],outputs=[input_text, result_text])
            clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text])
            doc_inputs.upload(fn=generate_file,inputs=[doc_inputs], outputs=[doc_outputs])
            
            print(gr.__version__)
    
        demo.queue(concurrency_count=10)
        demo.launch()


if __name__ == '__main__':
    main()