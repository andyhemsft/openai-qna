import os, logging 

# Flask modules
from flask import url_for, redirect

# App modules
from app import app

@app.route('/ui')
def hello_ui():
    '''
        Print out hello world
    '''
    return 'Hello World!'