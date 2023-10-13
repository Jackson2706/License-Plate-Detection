#!/bin/bash

python dataset/crawl_data.py
python clf_model.py
python main.py