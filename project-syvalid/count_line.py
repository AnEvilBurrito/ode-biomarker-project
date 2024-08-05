#!/usr/bin/env python

from json import load
from sys import argv
import os

print(os.getcwd())

def loc(nb):
    cells = load(open(nb))['cells']
    return sum(len(c['source']) for c in cells if c['cell_type'] == 'code')

def run(ipynb_files):
    return sum(loc(nb) for nb in ipynb_files)

if __name__ == '__main__':
    print(run(["project-syvalid/SYVALID_AN01_Curse_of_dimen.ipynb"]))
