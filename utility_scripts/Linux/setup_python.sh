#!/bin/bash
shopt -s expand_aliases
apt-get install python3.9
alias python=python3.9
apt install python3-pip
apt install python3.9-distutils
python -m pip install pip --upgrade pip
python -m pip install pydicom
python -m pip install pandas
python -m pip install numpy
python -m pip install tensorflow