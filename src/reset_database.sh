#!/bin/sh

rm register.db; touch register.db
python create_table.py
