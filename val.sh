#!/usr/bin/env bash
for dataset in raw asm contrast dissimilarity energy entropy homogeneity max mean std no2 nc5 he clahe lt gc lbp lbp_e lbp_f lbp_c; do
for model in x; do
    	time python val.py -d folds/5-Fold_$dataset/ -n fn_${dataset}_${model}_base
    done
done

rm -r `find . -name *val?`
