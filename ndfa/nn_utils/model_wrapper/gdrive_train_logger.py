import os
import json
import time
import pickle
import datetime
import tempfile
import subprocess
import dataclasses
from enum import Enum
import multiprocessing as mp
from typing import Optional, List, Tuple, Any

from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
from googleapiclient import errors


__all__ = ['GDriveTrainLogger']


class GDriveTrainLogger:
    def __init__(self, gdrive_base_folder_id: str, model_hps_hash: str, experiment_settings_hash: str):
        self.gdrive_base_folder_id = gdrive_base_folder_id
        self.train_folder_name = \
            f'model={model_hps_hash}__' \
            f'expr={experiment_settings_hash}__' \
            f'{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
        self.to_worker_msgs_queue = mp.Queue()
        self.background_process = mp.Process(
            target=_gdrive_background_worker,
            args=(self.gdrive_base_folder_id, self.train_folder_name, self.to_worker_msgs_queue,))
        self.background_process.start()

    def close(self):
        self.to_worker_msgs_queue.put(Action(action_kind=Action.ActionKind.Close), block=True)
        self.background_process.join()
        self.background_process.close()
        self.to_worker_msgs_queue.close()

    def run_subprocess_and_upload_stdout_as_text_file(
            self, subprocess_args: List[str], filename: str, ignore_fault: bool = False):
        self.to_worker_msgs_queue.put_nowait(Action(
            action_kind=Action.ActionKind.run_subprocess_and_upload_stdout_as_text_file,
            args=(subprocess_args, filename, ignore_fault)))

    def upload_string_as_text_file(self, text: str, filename: str):
        self.to_worker_msgs_queue.put_nowait(Action(
            action_kind=Action.ActionKind.upload_string_as_text_file,
            args=(text, filename)))

    def upload_as_json_file(self, json_dict: dict, filename: str):
        self.to_worker_msgs_queue.put_nowait(Action(
            action_kind=Action.ActionKind.upload_as_json_file,
            args=(json_dict, filename)))


@dataclasses.dataclass
class Action:
    class ActionKind(Enum):
        Close = 'Close'
        run_subprocess_and_upload_stdout_as_text_file = 'run_subprocess_and_upload_stdout_as_text_file'
        upload_string_as_text_file = 'upload_string_as_text_file'
        upload_as_json_file = 'upload_as_json_file'

    action_kind: ActionKind
    args: Tuple[Any, ...] = ()


def _gdrive_background_worker(
        gdrive_base_folder_id: int, train_folder_name: str, msgs_queue: mp.Queue):
    gdrive_logger = GDriveTrainLoggerBackgroundWorker(
        gdrive_base_folder_id=gdrive_base_folder_id, train_folder_name=train_folder_name)
    while True:
        pending_msg = msgs_queue.get(block=True)
        assert isinstance(pending_msg, Action)
        if pending_msg.action_kind == Action.ActionKind.Close:
            return
        elif pending_msg.action_kind == Action.ActionKind.run_subprocess_and_upload_stdout_as_text_file:
            gdrive_logger.run_subprocess_and_upload_stdout_as_text_file(*pending_msg.args)
        elif pending_msg.action_kind == Action.ActionKind.upload_string_as_text_file:
            gdrive_logger.upload_string_as_text_file(*pending_msg.args)
        elif pending_msg.action_kind == Action.ActionKind.upload_as_json_file:
            gdrive_logger.upload_as_json_file(*pending_msg.args)
        else:
            assert False


class GDriveTrainLoggerBackgroundWorker:
    def __init__(self, gdrive_base_folder_id: int, train_folder_name: str):
        self.gdrive_base_folder_id = gdrive_base_folder_id
        self.train_folder_name = train_folder_name
        self.gdrive_service = self._create_gdrive_service()
        self.train_folder_id = self._create_gdrive_folder(folder_name=self.train_folder_name)
        print(f'Logging to google drive folder '
              f'@ `https://drive.google.com/drive/u/folders/{self.train_folder_id}` '
              f'named `{self.train_folder_name}`.')
        self.filename_to_file_id_mapping = {}

    def run_subprocess_and_upload_stdout_as_text_file(
            self, subprocess_args: List[str], filename: str, ignore_fault: bool = False):
        if ignore_fault:
            try:
                with subprocess.Popen(
                        args=subprocess_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL) as process:
                    process_stdout = process.communicate()[0].decode("utf-8")
            except:
                return None
            return self.upload_string_as_text_file(process_stdout, filename)
        else:
            with subprocess.Popen(
                    args=subprocess_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL) as process:
                process_stdout = process.communicate()[0].decode("utf-8")
                return self.upload_string_as_text_file(process_stdout, filename)

    def upload_string_as_text_file(self, text: str, filename: str):
        with tempfile.NamedTemporaryFile('w') as file:
            file.write(text)
            file.seek(0)
            return self._update_file_to_train_folder(
                local_file_path=file.name,
                target_file_name=filename)

    def upload_as_json_file(self, json_dict: dict, filename: str):
        with tempfile.NamedTemporaryFile('w') as file:
            json.dump(json_dict, file)
            file.seek(0)
            return self._update_file_to_train_folder(
                local_file_path=file.name,
                target_file_name=filename)

    def _update_file_to_train_folder(
            self,
            local_file_path: str,
            target_file_name: Optional[str] = None,
            mimetype: str = 'text/plain;charset=UTF-8') -> Optional[str]:
        NR_ATTEMPTS = 4
        for attempt_nr in range(1, NR_ATTEMPTS + 1):
            try:
                target_file_name = os.path.basename(local_file_path) if target_file_name is None else target_file_name
                file_metadata = {
                    'name': target_file_name,
                    'parents': [self.train_folder_id]
                }

                if target_file_name in self.filename_to_file_id_mapping:
                    file_id = self.filename_to_file_id_mapping[target_file_name]
                    # we don't need loading the file, as we don't update it's metadata fields.
                    # file = self.gdrive_service.files().get(fileId=file_id).execute()
                    media = MediaFileUpload(
                        local_file_path, mimetype=mimetype)  # , resumable=True
                    updated_file = self.gdrive_service.files().update(
                        fileId=file_id,
                        # body=file,  # no need to update metadata fields
                        # newRevision=False,  # only valid param for API v2
                        media_body=media).execute()
                    return file_id
                else:
                    media = MediaFileUpload(local_file_path, mimetype=mimetype)  # , resumable=True
                    file = self.gdrive_service.files().create(
                        body=file_metadata, media_body=media, fields='id').execute()
                    file_id = file.get('id')
                    self.filename_to_file_id_mapping[target_file_name] = file_id
                    return file_id
            except errors.HttpError:
                if attempt_nr < NR_ATTEMPTS:
                    time.sleep(5 * attempt_nr)
        return None

    @classmethod
    def _get_gdrive_credentials(cls):
        credentials_file_path = 'credentials/gdrive_credentials.json'
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('credentials/gdrive_token.pickle'):
            with open('credentials/gdrive_token.pickle', 'rb') as token:
                try:
                    creds = pickle.load(token)
                except:
                    pass
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                creds = service_account.Credentials.from_service_account_file(
                    credentials_file_path, scopes=['https://www.googleapis.com/auth/drive'])
            # Save the credentials for the next run
            try:
                with open('credentials/gdrive_token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            except:
                if os.path.isfile('credentials/gdrive_token.pickle'):
                    os.remove('credentials/gdrive_token.pickle')
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
