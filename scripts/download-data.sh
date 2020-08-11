#! /bin/bash
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

if [ `uname -s` == "Darwin" ]; then
  DOWNLOADER="curl -o"
else
  DOWNLOADER="wget -O"
fi

data_dir=$1
mkdir -p $data_dir
cd $data_dir

if [ ! -d "results"  ]; then
  $DOWNLOADER results2016.zip https://pithos.okeanos.grnet.gr/public/Um13GYS9koXLg5Y6MrvBz2
  unzip results2016
fi

if [ ! -d "results2019"  ]; then
  $DOWNLOADER results2019.tar.gz https://pithos.okeanos.grnet.gr/public/RWT7pSP2OekiPHOvjtzuH7
  tar -xvf results2019.tar.gz
fi

