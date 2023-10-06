import pytest
import logging

from app.utils.file.storage import BlobStorageClient
from app.config import Config

logger = logging.getLogger(__name__)

@pytest.fixture
def blob_storage_client():
    """This fixture returns a BlobStorageClient object."""
    config = Config()
    return BlobStorageClient(config=config)


def test_extract_container_blob_name(blob_storage_client):
    """This function tests the _extract_container_blob_name function."""

    url = f'https://{Config().BLOB_ACCOUNT_NAME}.blob.core.windows.net/test/test_folder/test.txt'
    container, blob_name = blob_storage_client._extract_container_blob_name(url=url)
    assert container == 'test'
    assert blob_name == 'test_folder/test.txt'

def test_get_blob_sas(blob_storage_client):
    """This function tests the get_sas_url function."""

    url = f'https://{Config().BLOB_ACCOUNT_NAME}.blob.core.windows.net/test/test_folder/test.txt'
    container, blob_name = blob_storage_client._extract_container_blob_name(url=url)
    sas_url = blob_storage_client.get_blob_sas(container, blob_name)