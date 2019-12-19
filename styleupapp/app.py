import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import os
import cv2
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn.cluster import KMeans
import sys
import argparse
from base64 import b64encode
from flask import Flask, Response
import pickle


#plt.style.use('seaborn-white')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

app.scripts.config.serve_locally = True

styleup = './static/styleup.jpg'

app.layout = html.Div([
    html.Img(src='data:image/jpg;base64,{}'.format(b64encode(open('{}'.format(styleup), 'rb').read()).decode())),
    html.Label('Type of Clothing'),
                       dcc.Dropdown(
                                id='style',
                                options=[{'label': 'Long Sleeve', 'value': 'long_sleeve'},
                                {'label': 'Short Sleeve', 'value': 'short_sleeve'},
                                {'label': 'Sleeveless', 'value': 'sleeveless'}],
                                    value='upper_clothing'
                                    ),

                       dcc.Upload(
                                  id='upload-image',
                                  children=html.Div([
                                                     'Drag and Drop or ',
                                                     html.A('Select Files')
                                                     ]),
                                  style={
                                  'width': '100%',
                                  'height': '60px',
                                  'lineHeight': '60px',
                                  'borderWidth': '1px',
                                  'borderStyle': 'dashed',
                                  'borderRadius': '5px',
                                  'textAlign': 'center',
                                  'margin': '10px'
                                  },
                                  # Allow multiple files to be uploaded
                                  multiple=True
                                  ),
                       html.Div(id='output-image-upload'),
                       html.Div(id='output-matching-image-upload'),
                       ])

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr()
        ])


@app.callback(Output('output-image-upload','children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
                    parse_contents(c, n, d) for c, n, d in
                    zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#app.callback(Output('intermediate-value', 'children'), [Input('upload-image', 'children')])
def links_func(link):
    return html.Div([
                     html.H5("Matching clothes"),
                     html.A('Link', href='{}'.format(link), target='_blank'),
                     #html.Img(src='data:image/jpeg;base64,{}'.format(name)),
                     html.Hr()
                ])
def parse_names(name):
    return html.Div([
                     html.H5("Matching clothes"),
                     #html.H5('{}'.format(name)),
                     # HTML images accept base64 encoded strings in the same format
                     # that is supplied by the upload
                     html.Img(src='data:image/jpeg;base64,{}'.format(name)),
                     html.Hr()
                     ])
@app.callback(Output('output-matching-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def img_label_display(list_of_contents, list_of_names):
    MODEL_PATH = 'models/img_cluster.pkl'
    img_path = './static/images/'
    model = pickle.load(open(MODEL_PATH,'rb'))
    l_image = []
    if list_of_names is not None:
        preds = model.predict(list_of_names[0], None, num_choice = 3, open_url = False)
        l_image = preds[0]
        l_link = preds[1]
        encoded_image = [b64encode(open(os.path.join(img_path, l_image[i]), 'rb').read()).decode() for i in range(len(l_image))]
        #print(l_image)
        #encoded_image = [b64encode(open(l_link[i], 'rb').read()).decode() for i in range(len(l_link))]
        #print(encoded_image)
        children = [html.Div([links_func(u) for u in l_link]),html.Div([parse_names(encoded_image[i]) for i in range(len(encoded_image))])]
        return children
                       
if __name__ == '__main__':
    app.run_server(debug=True)
