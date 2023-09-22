import logging
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

from app.config import Config

logger = logging.getLogger(__name__)

class FormRecognizer:

    def __init__(self, config: Config):

        self.config = config
        self.client = DocumentAnalysisClient(
            endpoint=config.FORM_RECOGNIZER_ENDPOINT, 
            credential=AzureKeyCredential(config.FORM_RECOGNIZER_KEY)
        )

    def analyze_document(self, file_path: str) -> dict:
        """This function analyzes a document using the Form Recognizer API.
        
        Args:
            file_path: the file path
        Returns:
            the analysis result
        """

        with open(file_path, "rb") as f:
            poller = self.client.begin_analyze_document(model_id="prebuilt-layout", document=f)
            layout = poller.result()


        layout = poller.result()

        for t in layout.tables:
            logger.info("Table:")
            logger.info(t.to_dict())
            
        return layout