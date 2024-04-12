#!/bin/sh

apt install python3-dev libavutil-dev libavformat-dev libswscale-dev
pip install timm


# get the coviar data loader
cd data coviar/data_loader/ && ./install.sh
