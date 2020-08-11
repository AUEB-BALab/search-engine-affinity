#! /bin/bash

git clone https://github.com/theosotr/py-ms-cognitive bing-search
cd bing-search
git checkout fix-link
python setup.py install
cd ../..
python setup.py install
cd scripts
pip install -r requirements.txt
rm bing-search -r -f
