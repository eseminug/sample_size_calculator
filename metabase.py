import requests
import pandas as pd
from typing import Any
from pydantic import BaseModel, Field
from json import dumps



class Mb_Client(BaseModel):
    url: str
    username: str
    password: str = Field(repr=False)
    session_header: dict = Field(default_factory=dict, repr=False)


    # def model_post_init(self, __context: Any) -> None:
    #     self.get_session()


    # def get_session(self) -> int:
    #     credentials: dict = {
    #         "username": self.username,
    #         "password": self.password
    #     }
        
    #     response = requests.post(
    #         f"{self.url}/api/session",
    #         json=credentials
    #     )
    #     if response.status_code != 200:
    #         return response.status_code

    #     session_id = response.json()["id"]
    #     setattr(self, "session_header", {"X-Metabase-Session": session_id})
    #     return 200

    # def login(self, username, password) -> bool:
    #     """
    #     Login to Metabase and set the session header.
    #     """
    #     self.username = username
    #     self.password = password
    #     return_code = self.get_session()
    #     if return_code == 401:
    #         return False
    #     return True

    def post(self, api_endpoint: str, query: str, api_key) -> dict:
        payload: dict = {
            "database": 2,
            "type": "native",
            "format_rows": False,
            "pretty": False,
            "native": {
                "query": query
            }
        }
        post = requests.post(
            f'{self.url}/api/{api_endpoint}',
            headers = self.session_header | {
                "X-API-KEY": api_key,
                "Content-Type": "application/json;charset=utf-8"
            },
            json=payload
        )
        json_res = post.json()
        column_names = [col['display_name'] for col in json_res['data']['cols']]
        data_rows = json_res['data']['rows']
        df = pd.DataFrame(data_rows, columns=column_names)
        return df
