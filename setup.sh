#!/bin/sh

apt --assume-yes install python3-dev libavutil-dev libavformat-dev libswscale-dev unrar
pip install timm


# get the coviar data loader
cd pycoviar/data_loader/ && ./install.sh
