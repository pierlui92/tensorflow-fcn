#!/bin/bash
for i in 18000 23000 27000 32000 36000 41000 45000 50000 54000 59000 63000 72000 77000 81000 86000
do
 python3 test_fcn8_vgg.py --input_list_val_test filelist/input_list_val_test_cityscapes.txt --checkpoint_path checkpoint/fcn8s-$i --test_dir ./test_$i
 echo $i >> res.txt
 python3 eval.py --pred_folder ./test_$i --gt_file filelist/input_list_val_test_cityscapes.txt >> res.txt
done
