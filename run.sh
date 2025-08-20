#!/bin/bash

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# run script
python -m rag_qa.main