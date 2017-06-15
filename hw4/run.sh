#!/bin/bash
git clone https://Yusics@gitlab.com/Yusics/RL.git
git clone https://Yusics@gitlab.com/Yusics/S2S.git
python2 test_seq2seq.py $1 $2 $3
