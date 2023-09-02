import os
import logging
from abc import abstractmethod

from langchain.document_loaders import PyPDFLoader
from app.utils.file.blobstorage import BlobStorageClient
from app.config import Config

class Parser():
    """This class represents a Parser."""

    def __init__(self, config: Config):
        """
        Initialize the Parser.

        Args:
            config: the config object
        """

        self.config = config

    @abstractmethod
    def analyze_read(self, source_url: str) -> str:
        """
        Analyze and read source url.

        Args:
            source_url: the source url
        Returns:
            the parsed text
        """
        

    def write(self, dest_url: str, text: str):
        """
        Write the parsed text to a file.

        Args:
            dest_url: the destination url
            text: the parsed text
        """

        # Write to local file
        if self.config.DOCUMENT_DEST_LOCATION == 'local':
            with open(dest_url, 'w', encoding='utf-8') as f:
                f.write(text)

        # Write to Azure Blob Storage
        elif self.config.DOCUMENT_DEST_LOCATION == 'azure':
            blob_storage_client = BlobStorageClient(self.config)
            blob_storage_client.upload_blob(text, dest_url)

        return None


class PDFParser(Parser):
    """This class represents a PDF Parser."""

    def __init__(self, config: Config):
        """
        Initialize the PDF Parser.

        Args:
            config: the config object
        """

        self.config = config

        if self.config.PDF_PARSER_TYPE == 'pdfloader':
            pass

        elif self.config.PDF_PARSER_TYPE == 'formrecognizer':
            raise ValueError('Form Recognizer not supported yet')

        else:
            raise ValueError('PDF Parser type not supported')

    def analyze_read(self, source_url: str) -> str:
        """
        Analyze and read source url for a PDF.

        Args:
            source_url: the source url
        Returns:
            the parsed text
        """

        # Load PDF using langchain pdfloader
        if self.config.PDF_PARSER_TYPE == 'pdfloader':
            self.loader = PyPDFLoader(source_url)
            pages = self.loader.load()
  
            # Convert to UTF-8 encoding for non-ascii text
            for page in pages:
                try:
                    if page.page_content.encode("iso-8859-1") == page.page_content.encode("latin-1"):
                        page.page_content = page.page_content.encode("iso-8859-1").decode("utf-8", errors="ignore")
                except:
                    pass
            
            # Concatenate all pages
            text = '\n'.join([page.page_content for page in pages])

            return text

        elif self.config.PDF_PARSER_TYPE == 'formrecognizer':
            raise ValueError('Form Recognizer not supported yet')


