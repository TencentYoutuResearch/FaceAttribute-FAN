gt_file="../data/CelebA/list/test_full.txt"
pred_file="../result/test_full.txt"

python prediction_single_path.py --gpu="6" \
--attr-num="40" \
--prototxt-path="../outputs/deploy_single.prototxt" \
--caffemodel-path="../outputs/single_path_resnet_celeba.caffemodel" \
--test-file=${gt_file} \
--pred-file=${pred_file} \
--feature-layer="pred" \
--root-folder="../data/" \
--mean-file="../data/pretrained/ResNet_mean.binaryproto"

python cal_accuracy.py --gt-file=${gt_file} --pred-file=${pred_file}