#!/bin/bash

IP=$1
USER=$2

if [ -z "$IP" ]
then
  echo "Install this repo on a remote machine"
  echo "Usage: remote-install.sh <IP address> [user]"
  exit 1
fi

if [ -z "$USER" ]
then
  USER="ubuntu"
fi

echo "Copying files to nama directory"
rsync -a -e "ssh -l $USER" --exclude=/data --exclude=/src.egg-info . $IP:nama
scp ~/.netrc.wandb $USER@$IP:.netrc

echo "Installing requirements into nama conda environment and copying data dir from s3"
ssh $USER@$IP <<ENDSSH
# sudo apt-get install make
cd nama
make create_environment
source activate nama
make requirements
python -m ipykernel install --user --name nama
nbstripout --install
ENDSSH
