import gradio as gr
import pandas as pd
import json

df = pd.read_json('./role_template.json') 

def update_df(df):
    # 更新DataFrame
    return df

def save_df(df):
    # 将DataFrame保存为json文件
    df_json = df.to_json(orient='records')
    with open('role_template.json', 'w',encoding='utf-8') as f:
        json.dump(df_json, f,ensure_ascii=False)
    return df

with gr.Blocks() as demo:  
    data_table = gr.Dataframe(value=df, show_input=True, interactive=True)  
    save_button = gr.Button("Save")
    save_button.click(fn=save_df, inputs=[data_table], outputs=[data_table])
    data_table.change(fn=update_df, inputs=[data_table], outputs=[data_table])
    
demo.launch(share=True)