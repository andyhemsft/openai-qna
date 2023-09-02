import os

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, generate_container_sas, ContentSettings

from app.config import Config


class BlobStorageClient:

    def __init__(self, config: Config):
        """
        Initialize the Blob Storage Client.

        Args:
            config: the config object
        """

        self.config = config

        assert self.config.BLOB_ACCOUNT_NAME is not None, 'BLOB_ACCOUNT_NAME is not set in the environment variables'
        assert self.config.BLOB_ACCOUNT_KEY is not None, 'BLOB_ACCOUNT_KEY is not set in the environment variables'
        assert self.config.BLOB_CONTAINER_NAME is not None, 'BLOB_CONTAINER_NAME is not set in the environment variables'

        self.connection_string = 'DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net'.format(self.config.BLOB_ACCOUNT_NAME, self.config.BLOB_ACCOUNT_KEY)

        # Create the BlobServiceClient object which will be used to create a container client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

    def delete_blob(self, blob_name: str):
        """
        Delete a blob.

        Args:
            blob_name: the blob name
        """

        # Create a blob client using the blob name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=self.config.BLOB_CONTAINER_NAME, blob=blob_name)

        # Delete the blob
        blob_client.delete_blob()

    def upload_blob(self, data: str, blob_name: str):
        """
        Upload a blob.

        Args:
            data: the data to upload
            blob_name: the blob name
        """

        # Create a blob client using the blob name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=self.config.BLOB_CONTAINER_NAME, blob=file_path)

        # Upload the data
        blob_client.upload_blob(data, overwrite=True)