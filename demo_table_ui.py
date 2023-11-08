import gradio as gr
import pandas as pd
import json

df = pd.read_json('./role_template.json',orient='records') 

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

with gr.Blocks() as demo:
    data_table = gr.Dataframe(value=df, interactive=True)  
    save_button = gr.Button("Save")
    save_button.click(fn=save_df, inputs=[data_table], outputs=[data_table])
    data_table.change(fn=update_df, inputs=[data_table], outputs=None)
    
demo.launch(share=True)