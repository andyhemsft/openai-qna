import os

from flask import Flask
from app.utils.conversation.bot import LLMChatBot
from app.config import Config


# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

# Iinitialize the app
app = Flask(__name__)
app.static_folder = 'static'
app.config.from_object('app.config.Config')

# Import routing, models and Start the App
from app import models
from app.views import api, ui

