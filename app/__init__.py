import os

from flask import Flask


# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.from_object('app.config.Config')


# Import routing, models and Start the App
from app import views, models