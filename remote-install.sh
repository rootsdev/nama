#!/bin/bash

IP=$1

if [ -z "$IP" ]
then
  echo "Install this repo on a remote machine"
  echo "Usage: remote-install.sh <IP address>"
  exit 1
fi

echo "Copying files to nama directory"
rsync -a -e "ssh -l ubuntu" --exclude=data --exclude=data-raw --exclude=.git --exclude=src.egg-info . $IP:nama

echo "Installing requirements into nama conda environment and copying data dir from s3"
ssh ubuntu@$IP <<ENDSSH
cd nama
make create_environment
source activate nama
make requirements
make sync_data_from_s3
python -m ipykernel install --user --name nama
ENDSSH
