#!/usr/bin/env bash

for dataset in raw_clahe_std raw_lt_std raw_lbp_std lt_clahe_std lbp_mean_std; do
    for config in cfg/*; do
	    time python test.py -d folds/test/$dataset -n fr_${dataset}_x_`basename -s .yaml $config`
    done
done

rm -r `find . -name *val?`
