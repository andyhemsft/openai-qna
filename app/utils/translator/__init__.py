import logging

from app.config import Config

from azure.core.exceptions import HttpResponseError

import requests, uuid, json


logger = logging.getLogger(__name__)




class BaseTranslator:

    def __init__(self, config: Config) -> None:
        self.config = config


class AzureAITranslator(BaseTranslator):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Add your key and endpoint
        self.key = config.AZURE_TRANSLATOR_API_KEY
        self.endpoint = config.AZURE_TRANSLATOR_ENDPOINT

        # location, also known as region.
        # required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
        self.location = config.AZURE_TRANSLATOR_API_REGION

        self.path = '/translate'
        self.constructed_url = self.endpoint + self.path

        self.params = {
            'api-version': '3.0',
            'to': ['en']
        }

        self.headers = {
            'Ocp-Apim-Subscription-Key': self.key,
            # location required if you're using a multi-service or regional (not global) resource.
            'Ocp-Apim-Subscription-Region': self.location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def translate(self, text: str, target_language: str = "en") -> str:
        try:
            # You can pass more than one object in body.
            body = [{
                'text': text
            }]

            request = requests.post(self.constructed_url, params=self.params, headers=self.headers, json=body)
            response = request.json()

            logger.debug(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))

            return response[0]["translations"][0]["text"]
        
        except HttpResponseError as exception:
            logger.debug(f"Error Code: {exception.error.code}")
            logger.debug(f"Message: {exception.error.message}")


def get_translator(config: Config) -> BaseTranslator:
    """Get the translator based on the config.
    
    Args:
        config (Config): The config object.
    
    Returns:
        BaseTranslator: The translator object.
    """
    if config.OPENAI_API_TYPE == "azure":
        return AzureAITranslator(config)

    else:
        raise NotImplementedError(f"Translator {config.OPENAI_API_TYPE} is not implemented")