import os
import sys
import logging
import pytest

from app.utils.file import parser
from app.config import Config

logger = logging.getLogger(__name__)


@pytest.fixture()
def pdf_parser():
    config = Config()

    yield parser.PDFParser(config)

@pytest.fixture()
def excel_parser():
    config = Config()

    yield parser.ExcelParser(config)

def test_excel_parser(excel_parser):
    """Test the excel parser"""

    # sample_source_url = 'samples/AIA_POC.xlsx'
    sample_source_url = 'samples/hsbc_credit_card.xlsx'

    # Analyze and read source url for an Excel
    parsed_text = excel_parser.analyze_read(sample_source_url)

def test_langchain_pdf_parser(pdf_parser):
    """Test the langchain pdf parser"""

    sample_source_url = 'samples/gpt4_technical_report.pdf'

    # Analyze and read source url for a PDF
    parsed_text = pdf_parser.analyze_read(sample_source_url)

    assert 'GPT-4 Technical Report' in parsed_text

def test_extract_tables(pdf_parser):
    """Test the extract tables"""

    # sample_source_url = 'samples/hsbc_credit_card.pdf'
    sample_source_url = 'samples/gpt4_technical_report.pdf'

    # Extract tables
    table_text = pdf_parser._extract_tables(sample_source_url)

    logger.info(table_text)

    # assert 'Table 1' in table_text

def test_pdf_parser_write(pdf_parser):
    """Test the pdf parser write"""

    sample_source_url = 'samples/gpt4_technical_report.pdf'
    sample_dest_url = 'samples/gpt4_technical_report_test.txt'

    # Analyze and read source url for a PDF
    parsed_text = pdf_parser.analyze_read(sample_source_url)

    # Write the parsed text to a file
    pdf_parser.write(sample_dest_url, parsed_text)

    assert os.path.exists(sample_dest_url)

    # Remove the file
    os.remove(sample_dest_url)