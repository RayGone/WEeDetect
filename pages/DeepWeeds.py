import sys
sys.path.append('../models')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time

import dash
from dash import dcc, html
from dash import Input, Output, State 
from dash import ctx, clientside_callback, MATCH, ALL
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

dash.register_page(__name__, path='/', name='WEeDetect - DeepWeeds', description="Weeds Classification using Deep Learning")

models = getAvailableModels()
deep_weeds_labels = {0: 'Chinee apple', 1: 'Lantana', 8: 'Negative',
 2: 'Parkinsonia', 3: 'Parthenium', 4: 'Prickly acacia', 5: 'Rubber vine',
 6: 'Siam weed', 7: 'Snake weed'}

deep_weeds_labels_name = [deep_weeds_labels[i] for i in range(9)]
img_size_limit = int(os.getenv("IMG_SIZE_LIMIT", 5))


layout = dbc.Container([
        dbc.Row([            
            dbc.Col(width=12, children=[
                # dbc.Label([
                #     html.B("The model is trained using deepweeds dataset. Thus it classifies the weed image to one of "),
                #     html.Span(str(deep_weeds_labels_name)+".", className='text-sm d-block mb-2'),
                # ], className='text-start mb-2 text-xs'),
                dbc.Label([
                    html.B("Choose DL mode")
                ], className='text-start mb-2 text-xs'),
                dcc.RadioItems(
                    id='dw-model-select',
                    options=[
                        {'label':m['name'], 'value': m['name'].lower()} 
                        for m in models
                    ],
                    inline=True,
                    value=models[0]['name'].lower() if len(models) == 1 else [m['name'].lower() for m in models if m['isDefault']][0],
                    className='text-start mb-2 d-flex flex-row align-items-center justify-content-start g-2 overflow-auto',
                    labelClassName="me-2 form-check-label",
                    inputClassName="form-check-input me-2",
                    persistence=True,
                ),
                html.Details(className='text-start mb-2', children=[
                    html.Summary("Model Description", className='text-start mb-2'),
                    html.Div(id='model-summary', className='text-sm d-block mb-2'),
                ]),
                html.Div([
                    # html.Span("The model is trained on Kaggle and the notebook is available at: ", className='text-sm'),
                    # html.A("reganmaharjan/deepweeds-mobilenetv1/notebook",href="www.kaggle.com/code/reganmaharjan/deepweeds-mobilenetv1/notebook", target="_blank", title='Check Kaggle Notebook', className="text-sm", style={"cursor": "pointer"}),
                ]), html.Br(),
                html.H4("Upload weed images to classify.", className="text-start"),
                html.Span(f"Upload Image of size less than {img_size_limit}MB", className='text-sm text-start d-block mb-2'),
            ]),
        ], class_name='mb-3'),
        
        dbc.ButtonGroup([
            dbc.Button(id={"index":"toggle-btn", "value":"toggle-img-upload"}, children=html.I(className="fa fa-upload"), active=True),
            dbc.Button(id={"index":"toggle-btn", "value":"toggle-img-capture"}, children=html.I(className="fa fa-camera"))
        ], class_name="mb-2"),
        
        dbc.Row(
            id={'type': "dw-output-container", "id": "img-upload"},
            children = [
                dbc.Col(
                    dcc.Upload(id='upload-data',
                        max_size=img_size_limit * 1024 * 1024,  # 5MB
                        accept='image/*',
                        multiple=True,
                        children=html.Div([
                            dbc.Label('Drag and Drop or Select Files', style={"cursor":"pointer"}),
                        ]), 
                        className='w-100 text-center p-4 rounded-1 border-1',
                        style={
                            'lineHeight': '30px',
                            'borderStyle': 'dashed',
                        },
                        className_active='upload-dragdrop-active-bg'), sm=12),
                dbc.Col(
                    id='dw-output-display-container',
                    children=[
                        dbc.Card([
                            # dbc.CardHeader(id='image-upload-name', className='text-start'),
                            dbc.CardBody(
                                html.Div(id='dw-output-image-upload', 
                                        children=[
                                            dbc.Label("No Image Uploaded!!", id='dw-output-img-label', class_name="card-title text-start"), 
                                            html.Br(),
                                            dbc.CardImg(id='dw-output-image-upload-display',class_name='d-none w-sm-100 w-50', bottom=True)], 
                                        className='d-block w-100 text-center', style={"minHeight":"200px"}),
                            ),
                        ], class_name="w-100")
                    ], sm=12, class_name='text-center mt-3 w-100 d-flex flex-wrap pb-2')
        ], class_name='d-block'),
        
        dbc.Row(
            id={'type': "dw-output-container", "id": "img-capture"},
            children = [dbc.Col(
                dbc.Card([
                    # dbc.CardHeader(id='image-upload-name', className='text-start'),
                    dbc.CardBody(id="dw-cam"),
                    dcc.Store(id="img-capture-data"),
                    dbc.CardFooter([dbc.Button("Capture", id="img-catpure-trigger")])
                ], class_name="w-100"),
                sm=12, md=8, className='text-center mt-3'),
            dbc.Col(dbc.Card([
                dbc.CardImg(id="img-capture-display", class_name="w-25"),
                dcc.Loading(dbc.CardBody(id={"type":"container", "for": 'cam'}, 
                                         children=dbc.Label("Model Output:"), 
                                         class_name="text-start"),
                            type="circle")
            ]), sm=12, md=4, className='text-center mt-3')   
        ], class_name='d-none'),
       
    ], fluid=True)

@dash.callback(
    Output({"index":"toggle-btn", "value":ALL}, 'active'),
    Output({'type': "dw-output-container", "id": ALL}, 'class_name'),
    Input({"index":"toggle-btn", "value":ALL}, 'n_clicks'),
    State({"index":"toggle-btn", "value":ALL}, 'id'),
    prevent_initial_call=True,
)
def toggle_btn(n_clicks, ids):
    active = [id['value'] == ctx.triggered_id['value'] for id in ids]
    class_name = ["" if id['value'] == ctx.triggered_id['value'] else "d-none" for id in ids]
    
    return active, class_name
    # return tuple(active), tuple(class_name)

@dash.callback(
    Output('model-summary', 'children'),
    Input('dw-model-select', 'value'),
)    
def update_model_summary(model):
    model = [m for m in models if m['name'].lower() == model][0]
    summary = html.Div([html.Span(model['description']), html.Br(), 
                        html.Span(["Source: ", html.A(model['source_description'],href=model['source'], target="_blank", title='Check Source', className="text-sm", style={"cursor": "pointer"})])], className='text-sm d-block mb-2')
    
    return summary

##===============
def read_upload_image(contents):
    # Decode the base64 string and convert to an image
    header, encoded = contents.split(",", 1)  # Splits into "data:image/png;base64" and the base64 string
    image_data = base64.b64decode(encoded)
    
    # Convert to an Image object
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')  # Ensure image is in RGB format
    return image

def preprocess(images, model):
    config = [m for m in models if m['name'].lower() == model][0]
    images = [image.resize(config['input_shape']) for image in images]  # Resize to match model input size
    return np.array(images) / 255.0 # Normalize 0 and 1

def eval_model(image, model_name=None):  
    model = load_model(model_name)
    return model.predict(image, batch_size=image.shape[0])

##=========================
##-
@dash.callback(
    Output('dw-output-display-container', 'children'),
    Input('upload-data', 'filename'),
    Input('upload-data', 'contents')
)
def display_uploaded_images(filenames, contents):
    if not filenames:
        return dash.no_update
    
    children = []
    id = 0
    for name in filenames:
        child = dbc.Card([
                        dbc.CardHeader(name, id={'id':id, 'type': 'filename', "for":"upload"}, class_name='text-start text-nowrap', style={"fontSize":"0.8em"}),
                        dbc.CardImg(id={'id':id, 'type': 'image', "for":"upload"}, style={"maxWidth":"250px", "height":"150px"}, top=True, src=contents[id]),
                        dbc.CardBody(id={'id':id, "type":"container", "for":"upload"}, 
                            children=[dbc.Label("Model Output:"), html.Br(), html.I(className="fa fa-spinner fa-pulse text-primary", style={"fontSize": "24px"})], 
                            class_name="text-start", style={"font-size":'0.8em'}),                        
                    ], style={"width": "250px", "minHeight":"300px"}, class_name="p-2 m-1")
        children.append(child)
        id+=1
    return children

@dash.callback(
    Output({'id': ALL, "type":"container", "for":"upload"}, 'children', allow_duplicate=True),
    Input('dw-output-display-container', 'children'),
    State('upload-data', 'contents'),
    Input('dw-model-select', 'value'),
    prevent_initial_call=True,
)
def make_prediction_on_upload(_, contents, selected_model_name):
    time.sleep(1)
    if contents is not None:
        print("\nUpload Mode")
        print(f"Using Model: {selected_model_name} for prediction")
        images = [read_upload_image(content) for content in contents]
        images = preprocess(images, selected_model_name)
        prediction = eval_model(images, selected_model_name)
        outputs = []
        for p in prediction:
            top_k = tf.math.top_k(p, 2)
            topK_prob = list(top_k.values.numpy())
            topK_class = [deep_weeds_labels[idx] for idx in top_k.indices.numpy()]
            
            output = [html.H4("Model Output: ", className='card-title font-weight-bold', style={"fontSize":"1em"}), html.Hr(),
                *[
                    html.Div([
                        dbc.Label([html.B(f"Label: "), html.Span(f" {topK_class[i]}")], class_name="mb-1 w-100", style={"fontSize":"0.9em"}),
                        dbc.Label([html.B("Probability: "), html.Span(f" {topK_prob[i]:.4f}")], class_name="mb-1 w-100", style={"fontSize":"0.9em"}),
                        html.Br()
                    ], className="mb-2 border w-100 rounded-1 p-1")
                    for i in range(len(topK_prob))
                ]]
            outputs.append(output)
            
        return outputs
    
    return dash.no_update
        

@dash.callback(
    Output({"type":"container", "for":"cam"}, 'children', allow_duplicate=True),
    Input('img-capture-data', 'data'),
    Input('dw-model-select', 'value'),
    prevent_initial_call=True,
)
def make_prediction_on_cam_capture(capture, model):
    if capture is None or model is None:
        return dash.no_update
    
    contents = capture['image']
    
    print("\nCamera Mode") 
    print(f"Using Model: {model} for prediction")
    image = read_upload_image(contents)
    image = preprocess([image], model)
    prediction = eval_model(image, model)[0]
    top_k = tf.math.top_k(prediction, 2)
    topK_prob = list(top_k.values.numpy())
    topK_class = [deep_weeds_labels[idx] for idx in top_k.indices.numpy()]
    
    output = [html.H4("Model Output: ", className='card-title font-weight-bold'), html.Hr(),
            *[
                html.Div([
                    dbc.Label([html.B(f"Label: "), html.Span(f" {topK_class[i]}")], class_name="mb-1 w-100"),
                    dbc.Label([html.B("Probability: "), html.Span(f" {topK_prob[i]:.4f}")], class_name="mb-1 w-100"),
                    html.Br()
                ], className="mb-2 border w-100 rounded-1 p-1")
                for i in range(len(topK_prob))
            ]]
    # [html.Div(dbc.Label("File Name: "+filename, class_name="card-title text-start"), className='text-start'), dbc.CardImg(src=contents, class_name='w-50', bottom=True)], 
    return output

clientside_callback(
    """
    (active) => {
        const container = document.getElementById('dw-cam');
        
        const videoSrc = document.querySelector('video');
        
        if(videoSrc){
            const mediaStream = videoSrc.srcObject;
            if(mediaStream){
                const tracks = mediaStream.getTracks();
                tracks.forEach(track => track.stop());
                videoSrc.srcObject = null;
            }
        }
        
        container.innerHTML = ''
        
        
        const videoElement = document.createElement("video");
        videoElement.setAttribute('autoplay', '');
        videoElement.setAttribute('playsinline', '');
        videoElement.className = "w-50"
        if (active) {                
            try {
                container.appendChild(videoElement)
                window.navigator.mediaDevices.getUserMedia({ video: true })
                    .then((stream) =>{                        
                        videoElement.srcObject = stream;
                    });
            } catch (error) {
                console.error('Error accessing camera:', error);
            }
            return window.dash_clientside.no_update
        }
        return ""
    }
    """,
    Output('dw-cam', 'children'),
    Input({"index":"toggle-btn", "value":'toggle-img-capture'}, 'active'),
)

clientside_callback(
    """
    (click) => {
        const video = document.querySelector("video");
        
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        const imageData = canvas.toDataURL('image/png');
        return {"image": imageData}
    }
    """,
    Output('img-capture-data', 'data'),
    Input('img-catpure-trigger', 'n_clicks'),
    prevent_initial_call=True
)

clientside_callback(
    """
    (data) => {
        return data['image']
    }
    """,
    Output('img-capture-display', 'src'),
    Input('img-capture-data', 'data'),
    prevent_initial_call=True
)

##===========================================

# clientside_callback(
#     """
#     (filename) => {
#         if (filename) {
#             return "File Name: " + filename;
#         }
        
#         return "No Image Uploaded!!";
#     }
#     """,
#     Output('dw-output-img-label', 'children'),
#     Input('upload-data', 'filename'),
# )

# clientside_callback(
#     """
#     (filename, img_class) => {
#         if (filename) {
#             return "card-img-top w-50 w-sm-50";
#         }
        
#         return "d-none card-img-top w-50 w-sm-50";
#     }
#     """,
#     Output('dw-output-image-upload-display', 'class_name'),
#     Input('upload-data', 'filename'),
#     State('dw-output-image-upload-display', 'class_name'),
# )

# clientside_callback(
#     """
#     (contents) => {
#         if (contents) {
#             console.log(contents)
#             return contents;
#         }
        
#         return '';
#     }
#     """,
#     Output('dw-output-image-upload-display', 'src'),
#     Input('upload-data', 'contents'),
# )