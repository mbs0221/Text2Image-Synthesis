#!/bin/bash
root='coco-2014'
if [ ! -d $root ]; then
  mkdir -p $root
fi
while [ read line ]
do
  if [-n '$line']
  then
      url=$(echo '$line'|tr -d '\r')
      filename=$(echo ${url##*/})
      filepath=$(echo ${url%/*})
      echo $filename
      echo $filepath
  fi
done