import sys
sys.path.append('../models')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from plotly import express as px
import pandas as pd
import numpy as np

import base64
from io import BytesIO
from PIL import Image

# import torch
# from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
# from torchvision.transforms import transforms
import tensorflow as tf
from models import load_model
from utilities import getAvailableModels

dash.register_page(__name__, path='/', name='DeepWeeds', title="Weeds Classification", description="Weeds Classification using Deep Learning")

models = getAvailableModels()
deep_weeds_labels = {0: 'Chinee apple', 1: 'Lantana', 8: 'Negative',
 2: 'Parkinsonia', 3: 'Parthenium', 4: 'Prickly acacia', 5: 'Rubber vine',
 6: 'Siam weed', 7: 'Snake weed'}

deep_weeds_labels_name = [deep_weeds_labels[i] for i in range(9)]
img_size_limit = int(os.getenv("IMG_SIZE_LIMIT", 5))


layout = dbc.Container([
        dbc.Row([            
            dbc.Col(width=12, children=[
                dbc.Label([
                    html.B("The model is trained using deepweeds dataset. Thus it classifies the weed image to one of "),
                    html.Span(str(deep_weeds_labels_name)+".", className='text-sm d-block mb-2'),
                ], className='text-start mb-2 text-xs'),
                dcc.RadioItems(
                    id='dw-model-select',
                    options=[
                        {'label':m['name'], 'value': m['name'].lower()} 
                        for m in models
                    ],
                    inline=True,
                    value=models[0]['name'].lower() if len(models) == 1 else [m['name'].lower() for m in models if m['isDefault']][0],
                    className='text-start mb-2 d-flex flex-row align-items-center justify-content-start g-2',
                    labelClassName="me-2 form-check-label",
                    persistence=True,
                ),
                html.Details(className='text-start mb-2', children=[
                    html.Summary("Model Description", className='text-start mb-2'),
                    html.P(id='model-summary', className='text-sm d-block mb-2'),
                ]),
                html.Div([
                    # html.Span("The model is trained on Kaggle and the notebook is available at: ", className='text-sm'),
                    # html.A("reganmaharjan/deepweeds-mobilenetv1/notebook",href="www.kaggle.com/code/reganmaharjan/deepweeds-mobilenetv1/notebook", target="_blank", title='Check Kaggle Notebook', className="text-sm", style={"cursor": "pointer"}),
                ]), html.Br(),
                html.H4("Upload an weed image to classify it using a pre-trained model.", className="text-center"),
                html.Span(f"Upload Image of size less than {img_size_limit}MB", className='text-sm text-start d-block mb-2'),
            ]),
        ], class_name='mb-3'),
        dcc.Upload(id='upload-data',
            max_size=img_size_limit * 1024 * 1024,  # 5MB
            accept='image/*',
            children=html.Div([
                dbc.Label('Drag and Drop or Select Files', style={"cursor":"pointer"}),
            ]), 
            className='w-100 text-center p-4 rounded-1 border-1',
            style={
                'lineHeight': '30px',
                'borderStyle': 'dashed',
            },
            className_active='upload-dragdrop-active-bg'),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    # dbc.CardHeader(id='image-upload-name', className='text-start'),
                    dbc.CardBody(dcc.Loading(html.Div(id='dw-output-image-upload', children=[dbc.Label("No Image Uploaded!!")], className='w-100 text-center', style={"minHeight":"200px"}), type="circle")),
                ], class_name="w-100"),
                sm=12, md=8, className='text-center mt-3'),
            dbc.Col(dbc.Card(
                dbc.CardBody(id='dw-model-output-container', children=dbc.Label("Model Output:"), class_name="text-start")),sm=12, md=4, className='text-center mt-3')
        ]),
       
    ], fluid=True)

def read_upload_image(contents):
    # Decode the base64 string and convert to an image
    header, encoded = contents.split(",", 1)  # Splits into "data:image/png;base64" and the base64 string
    image_data = base64.b64decode(encoded)
    
    # Convert to an Image object
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')  # Ensure image is in RGB format
    return image

def eval_model(image: Image.Image, model_name=None):  
    config = [m for m in models if m['name'].lower() == model_name][0]
    model = load_model(model_name)
    
    # Step 3: Resize and normalize the image
    image = image.resize(config['input_shape'])  # Resize to match model input size
    
    # image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W) format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    if config['model_framework'] == 'PyTorch':
        image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert to tensor
    
    if model_name == 'mobilenet':
        return model.predict(image)
    else:
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        return model.predict(image, batch_size=1)

@dash.callback(
    Output('model-summary', 'children'),
    Input('dw-model-select', 'value'),
)    
def update_model_summary(model):
    model = [m for m in models if m['name'].lower() == model][0]
    summary = html.Div([html.Span(model['description']), html.Br(), 
                        html.Span(["Source: ", html.A(model['source_description'],href=model['source'], target="_blank", title='Check Source', className="text-sm", style={"cursor": "pointer"})])], className='text-sm d-block mb-2')
    
    return summary

@dash.callback(
    Output('dw-output-image-upload', 'children'),
    Output('dw-model-output-container', 'children'),
    # Input('upload-data-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('dw-model-select', 'value'),
    prevent_initial_call=True,
)
def update_output( contents, filename, model):
    print(f"\n\nUsing Model: {model} for prediction")
    if contents is not None: 
        image = read_upload_image(contents)
        prediction = eval_model(image, model)[0]
        print(prediction)
        idx = np.argmax(prediction)
        topK_prob = [prediction[idx]]
        topK_class = [deep_weeds_labels[idx]]
        
        output = [html.H4("Model Output: ", className='card-title font-weight-bold'), html.Hr(),
                *[
                    html.P([
                        dbc.Label([html.B(f"Label: "), html.Span(f" {topK_class[i]}")], class_name="mb-1 w-100"),
                        dbc.Label([html.B("Probability: "), html.Span(f" {topK_prob[i]:.4f}")], class_name="mb-1 w-100"),
                        html.Br()
                    ], className="mb-2 border w-100 rounded-1 p-1")
                    for i in range(len(topK_prob))
                ]]
        return [html.Div(dbc.Label("File Name: "+filename, class_name="card-title text-start"), className='text-start'), dbc.CardImg(src=contents, class_name='w-50', bottom=True)], output
        
    return dash.no_update, dash.no_update