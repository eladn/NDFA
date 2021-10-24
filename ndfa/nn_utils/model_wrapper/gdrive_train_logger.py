import os
import json
import pickle
import datetime
import tempfile
import subprocess
from typing import Optional, List
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload


__all__ = ['GDriveTrainLogger']


class GDriveTrainLogger:
    def __init__(self, gdrive_folder_id: str, model_hps_hash: str, experiment_settings_hash: str):
        self.gdrive_base_folder_id = gdrive_folder_id
        self.gdrive_service = self._create_gdrive_service()
        self.train_folder_name = \
            f'model={model_hps_hash}__' \
            f'expr={experiment_settings_hash}__' \
            f'{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
        self.train_folder_id = self._create_gdrive_folder(folder_name=self.train_folder_name)

    def run_subprocess_and_upload_stdout_as_text_file(self, subprocess_args: List[str], filename: str):
        with subprocess.Popen(
                args=subprocess_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL) as process:
            process_stdout = process.communicate()[0].decode("utf-8")
            self.upload_string_as_text_file(process_stdout, filename)

    def upload_string_as_text_file(self, text: str, filename: str):
        with tempfile.NamedTemporaryFile('w') as file:
            file.write(text)
            file.seek(0)
            self._update_file_to_train_folder(
                local_file_path=file.name,
                target_file_name=filename)

    def upload_as_json_file(self, json_dict: dict, filename: str):
        with tempfile.NamedTemporaryFile('w') as file:
            json.dump(json_dict, file)
            file.seek(0)
            self._update_file_to_train_folder(
                local_file_path=file.name,
                target_file_name=filename)

    def _update_file_to_train_folder(
            self,
            local_file_path: str,
            target_file_name: Optional[str] = None,
            mimetype: str = 'text/plain;charset=UTF-8') -> str:
        target_file_name = os.path.basename(local_file_path) if target_file_name is None else target_file_name
        file_metadata = {
            'name': target_file_name,
            'parents': [self.train_folder_id]
        }
        media = MediaFileUpload(local_file_path, mimetype=mimetype)
        file = self.gdrive_service.files().create(
            body=file_metadata, media_body=media, fields='id').execute()
        file_id = file.get('id')
        return file_id

    @classmethod
    def _get_gdrive_credentials(cls):
        credentials_file_path = 'credentials/gdrive_credentials.json'
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('credentials/gdrive_token.pickle'):
            with open('credentials/gdrive_token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                creds = service_account.Credentials.from_service_account_file(
                    credentials_file_path, scopes=['https://www.googleapis.com/auth/drive'])
            # Save the credentials for the next run
            with open('credentials/gdrive_token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        return creds

    @classmethod
    def _create_gdrive_service(cls):
        return build('drive', 'v3', credentials=cls._get_gdrive_credentials())

    def _create_gdrive_folder(self, folder_name: str) -> str:
        file_metadata = {
            'parents': [self.gdrive_base_folder_id],
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'}
        file = self.gdrive_service.files().create(
            body=file_metadata, fields='id').execute()
        folder_id = file.get('id')
        return folder_id
