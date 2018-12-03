#!/bin/bash
#
#    Script - convert tif to png
#
#    Name: script_convertall.sh
#    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
#

echo "[SCRIPT CONVERT ALL] Initializing..."

dir_train="../data_daninhas/train"
dir_validation="../data_daninhas/validation"

echo "[SCRIPT CONVERT ALL] Converting train..."

for dir_class in `ls $dir_train`;
do
    echo "[SCRIPT CONVERT ALL] Converting class -" $dir_class;
    convert $dir_train/$dir_class/* $dir_train/$dir_class/$dir_class.png
    echo "[SCRIPT CONVERT ALL] Removing all .tif files in $dir_class ..."
    rm $dir_train/$dir_class/*.tif
done

echo "[SCRIPT CONVERT ALL] Converting validation..."

for dir_class in `ls $dir_validation`;
do
    echo "[SCRIPT CONVERT ALL] Converting class -" $dir_class;
    convert $dir_validation/$dir_class/* $dir_validation/$dir_class/$dir_class.png
    echo "[SCRIPT CONVERT ALL] Removing all .tif files in $dir_class ..."
    rm $dir_validation/$dir_class/*.tif
done

echo "[SCRIPT CONVERT ALL] OK! DONE."
