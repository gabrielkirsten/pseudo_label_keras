#!/bin/bash
#
#    Script - split data between train and test
#
#    Name: script_split_data.sh
#    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
#

echo "[SCRIPT SPLIT DATA] Initializing..."

dir_train="../data_daninhas/train"
dir_validation="../data_daninhas/validation"
dir_test="../data_daninhas/test"
perc_train=60
perc_validation=20
perc_test=20

for dir_class in `ls $dir_train`;
do
    echo "[SCRIPT SPLIT DATA] Spliting class -" $dir_class;
    mkdir $dir_validation/$dir_class
    mkdir $dir_test/$dir_class
    quantity_files=`ls $dir_train/$dir_class | wc -l`
    perc_quantity_files_train=$((($quantity_files/100)*$perc_train))
    perc_quantity_files_test=$(($quantity_files-(($quantity_files/100)*$perc_test)))
    counter=0
    arrayFiles=`ls $dir_train/$dir_class |sort -R`
    for file in $arrayFiles;
    do
        let "counter += 1"
        if [[ $counter -gt $perc_quantity_files_train && $counter -lt $perc_quantity_files_test ]]; then
            mv $dir_train/$dir_class/$file $dir_validation/$dir_class/$file
        fi
        if [[ $counter -gt $perc_quantity_files_test ]]; then
            mv $dir_train/$dir_class/$file $dir_test/$dir_class/$file
        fi
    done
done

echo "[SCRIPT SPLIT DATA] OK! DONE."
