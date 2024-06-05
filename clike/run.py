import os
from pickle import dump
import IPython
import subprocess

from .execute import aims

def run(aim: aims, *args):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/cash/arguments.pkl", 'wb') as file:
        dump(args, file)
    if IPython.get_ipython():
        print("Running: Jupyter Notebook")
        print(f"Trying to run: {os.path.dirname(os.path.abspath(__file__))}/execute.py '{aim}'")
        IPython.get_ipython().system(f'python3 "{os.path.dirname(os.path.abspath(__file__))}"/execute.py "{aim}"')
    else:
        print("Running: Direct")
        subprocess.run(["python3", f"{os.path.dirname(os.path.abspath(__file__))}/execute.py", str(aim)])