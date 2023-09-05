import os
import sys
import logging

from app.utils.file import parser
from app.config import Config

logger = logging.getLogger(__name__)

def test_langchain_pdf_parser():
    """Test the langchain pdf parser"""

    # Load config
    config = Config()

    sample_source_url = 'samples/gpt4_technical_report.pdf'

    # Load PDF using langchain pdfloader
    pdf_parser = parser.PDFParser(config)

    # Analyze and read source url for a PDF
    parsed_text = pdf_parser.analyze_read(sample_source_url)

    assert 'GPT-4 Technical Report' in parsed_text


def test_pdf_parser_write():
    """Test the pdf parser write"""

    # Load config
    config = Config()

    sample_source_url = 'samples/gpt4_technical_report.pdf'
    sample_dest_url = 'samples/gpt4_technical_report_test.txt'

    # Load PDF using langchain pdfloader
    pdf_parser = parser.PDFParser(config)

    # Analyze and read source url for a PDF
    parsed_text = pdf_parser.analyze_read(sample_source_url)

    # Write the parsed text to a file
    pdf_parser.write(sample_dest_url, parsed_text)

    assert os.path.exists(sample_dest_url)

    # Remove the file
    os.remove(sample_dest_url)