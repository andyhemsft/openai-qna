import os
from datetime import datetime, timedelta

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
        # assert self.config.BLOB_CONTAINER_NAME is not None, 'BLOB_CONTAINER_NAME is not set in the environment variables'

        self.connection_string = f'DefaultEndpointsProtocol=https;AccountName={self.config.BLOB_ACCOUNT_NAME};AccountKey={self.config.BLOB_ACCOUNT_KEY};EndpointSuffix=core.windows.net'

        # Create the BlobServiceClient object which will be used to create a container client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

    def delete_blob(self, container: str, blob_name: str) -> None:
        """
        Delete a blob.

        Args:
            blob_name: the blob name
        """

        # Create a blob client using the blob name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=container, blob=blob_name)

        # Delete the blob
        blob_client.delete_blob()

    def upload_blob(self, data: str, container: str, blob_name: str) -> None:
        """
        Upload a blob.

        Args:
            data: the data to upload
            blob_name: the blob name
        """

        # Create a blob client using the blob name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=container, blob=blob_name)

        # Upload the data
        blob_client.upload_blob(data, overwrite=True)

    def get_blob_sas(self, container: str, blob_name: str) -> str:
        """
        Get a SAS token for a blob.

        Args:
            blob_name: the blob name
        Returns:
            the SAS URL
        """

        return f"https://{self.config.BLOB_ACCOUNT_NAME}.blob.core.windows.net/{self.container_name}/{blob_name}" \
            + "?" + generate_blob_sas(
                        account_name=self.config.BLOB_ACCOUNT_NAME, 
                        container_name=container, 
                        blob_name=blob_name, 
                        account_key= self.config.BLOB_ACCOUNT_KEY, 
                        permission='r', 
                        expiry=datetime.utcnow() + timedelta(hours=1)
                    )