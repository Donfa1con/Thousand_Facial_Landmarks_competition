#!/bin/bash
# https://drive.google.com/open?id=1fD9BlL9VEc0WC3ybRWc1LRZKfYHftR4G dataset link

fileid="1fD9BlL9VEc0WC3ybRWc1LRZKfYHftR4G"
filename="dataset.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

rm ./cookie
unzip dataset.zip -d ../