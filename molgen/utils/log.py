import os
from .compute import is_master_rank

def mprint(string:str):
    if is_master_rank(): print(string)