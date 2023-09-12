import os, logging 

# Flask modules
from flask import render_template

# App modules
from app import app

@app.route('/ui')
def ui_home():
    """Render the UI home page"""

    return render_template('index.html')