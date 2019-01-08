import numpy as np
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Attribute Network")
    parser.add_argument("--gt-file", type=str,
                        help="ground truth file path.")
    parser.add_argument("--pred-file", type=str,
                        help="prediction file path.")
    parser.add_argument("--path-shift", type=int, default=0,
                        help="prediction file path.")
    return parser.parse_args()


def cal_attr():
    args = get_arguments()
    path_shift = args.path_shift
    print(args)
    gt_attr_file = args.gt_file
    pred_file = args.pred_file
    attr_num = 40
    gt_f = open(gt_attr_file, 'r')
    gt_line = gt_f.readline().strip().split()
    pred_f = open(pred_file, 'r')
    pred_line = pred_f.readline().strip().split()
    same_count = np.zeros(attr_num, dtype=np.int32)
    valid_sum = 0
    while pred_line:
        valid_sum += 1
        for i in range(attr_num):
            pred_attr = int(pred_line[1 + i])
            gt_attr = int(gt_line[path_shift + 1 + i])
            if pred_attr == gt_attr:
                same_count[i] += 1
        gt_line = gt_f.readline().strip().split()
        pred_line = pred_f.readline().strip().split()
    print(valid_sum)
    result = np.zeros(attr_num)
    cur_index = 0
    for v in same_count:
        print(v * 1.0 / valid_sum * 100)
        result[cur_index] = v * 1.0 / valid_sum * 100
        cur_index += 1
    print('mean result', np.mean(result))
    return result


if __name__ == '__main__':
    cal_attr()
