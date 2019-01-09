gt_file="../data/CelebA/list/test_with_abstraction_full.txt"
pred_file="../result/test_with_abstraction_full.txt"

python prediction_dual_path.py --gpu="4" \
--attr-num="40" \
--prototxt-path="../outputs/deploy_dual.prototxt" \
--caffemodel-path="../outputs/dual_path_parse_resnet_celeba.caffemodel" \
--test-file=${gt_file} \
--pred-file=${pred_file} \
--feature-layer="pred" \
--root-folder="../data/" \
--mean-file="../data/pretrained/ResNet_mean.binaryproto"

python cal_accuracy.py --gt-file=${gt_file} --pred-file=${pred_file} --path-shift=1