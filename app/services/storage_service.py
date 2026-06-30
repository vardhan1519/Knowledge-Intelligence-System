import boto3
from botocore.exceptions import ClientError
from app.config import Config

class S3StorageService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            #region_name=Config.AWS_REGION
        )
        self.bucket_name = Config.AWS_BUCKET_NAME

    def upload_file(self, file_path: str, object_name: str) -> bool:
        """Upload a file to S3 bucket."""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False

    def download_file(self, object_name: str, file_path: str) -> bool:
        """Download a file from S3 bucket."""
        try:
            self.s3_client.download_file(self.bucket_name, object_name, file_path)
            return True
        except ClientError as e:
            print(f"Error downloading file: {e}")
            return False

    def delete_file(self, object_name: str) -> bool:
        """Delete a file from S3 bucket."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError as e:
            print(f"Error deleting file: {e}")
            return False