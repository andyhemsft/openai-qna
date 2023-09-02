import os, logging 

# Flask modules
from flask import url_for, redirect

# App modules
from app import app

# App main route + generic routing
@app.route('/')
def index():
    '''
        Print out hello world
    '''
    return 'Hello World!'

