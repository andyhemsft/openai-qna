import logging
import pytest

from app.config import Config
from app.utils.file.formrecognizer import FormRecognizer


logger = logging.getLogger(__name__)


# TODO: need to mock the form recognizer API
# @pytest.fixture()
# def form_recognizer():
#     config = Config()

#     yield FormRecognizer(config)


# def test_form_recognizer(form_recognizer):
#     file_path = "samples/hsbc_credit_card.pdf"
#     form_recognizer.analyze_document(file_path)