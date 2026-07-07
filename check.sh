#!/bin/bash

expected=10
error=0

for i in runs/detect/*; do
    count=`ls $i | wc -l`
    if [ $count != $expected ]; then
        echo -e "\e[0;31mfolder $i: $count\e[0;39m"
        error=1
    else
        echo "folder $i: $count"
    fi
done

echo `ls runs/detect | wc -l` folders checked

if [ $error == 1 ]; then
    echo -e "\e[0;31mOne or more experiments have failed\e[0;39m"
    exit 1
else
    echo -e "\e[0;32mFinished succesfully\e[0;39m"
    exit 0
fi

