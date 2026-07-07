for method in raw asm contrast dissimilarity energy entropy homogeneity max mean std lbp lbp_e lbp_f lbp_c no2 nc5 he clahe lt gc raw_clahe_lt raw_max_std raw_no2_nc5 raw_clahe_std raw_lt_std raw_lbp_std lt_clahe_he lt_clahe_std lbp_mean_std lbp_max_std; do
    for i in {1..5}; do
        mkdir -p folds/train/$method/split_${i}/train/images
        mkdir -p folds/train/$method/split_${i}/train/labels

        mkdir -p folds/train/$method/split_${i}/val/images
        mkdir -p folds/train/$method/split_${i}/val/labels

        cp dataset/train/data.yaml folds/train/$method/split_${i}/split_${i}_dataset.yaml
        sed -i -e "s/dataset/$method/" -i -e "s/split/split_${i}/" folds/train/$method/split_${i}/split_${i}_dataset.yaml
    done
    
    mkdir -p folds/test/$method/val/images
    mkdir -p folds/test/$method/val/labels

    cp dataset/test/labels/* folds/test/$method/val/labels

    cp dataset/test/data.yaml folds/test/$method/
    sed -i "s/dataset/$method/" folds/test/$method/data.yaml
done

mkdir -p metrics

for method in raw_clahe_lt raw_max_std raw_no2_nc5 raw_clahe_std raw_lt_std raw_lbp_std lt_clahe_he lt_clahe_std lbp_mean_std lbp_max_std; do
    sed -i '1s/^channels: 1$/channels: 3/' folds/train/${method}/split_*/split_*_dataset.yaml
    sed -i '1s/^channels: 1$/channels: 3/' folds/test/${method}/data.yaml
done