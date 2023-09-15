
import json
import os.path as osp
from typing import Union, List


class Prompter(object):

    def __init__(self, verbose: bool = False):
        self._verbose = verbose


    def generate_prompt(
        self,
        prefix: str,
        text: Union[None, str] = None,
    ) -> str:

        res = prefix
        if text:
            res = f"{res}{text}"
        if self._verbose:
            print(res)
        return res




