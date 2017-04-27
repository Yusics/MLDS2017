#!/bin/bash
git clone https://Yusics@gitlab.com/Yusics/hw2.git
python2 gen.py $1 $2
python2 testing.py
python2 postprocess.py