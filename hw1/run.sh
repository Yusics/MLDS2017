#!/bin/bash
git clone https://Yusics@gitlab.com/Yusics/hw1.git
python2 preprocess-test.py $1
python2 test-lstm-1hot.py $2