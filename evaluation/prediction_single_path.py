import os
import numpy as np
import caffe
import cv2
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Attribute Network")
    parser.add_argument("--test-file", type=str,
                        help="test file path.")
    parser.add_argument("--pred-file", type=str,
                        help="prediction file path.")
    parser.add_argument("--feature-layer", type=str,
                        help="the name of the attribute layer.")
    parser.add_argument("--root-folder", type=str, default='./data',
                        help="image data root folder.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="gpu id.")
    parser.add_argument("--attr-num", type=str, default='40',
                        help="total attribute num.")
    parser.add_argument("--mean-file", type=str, default='../data/pretrained/ResNet_mean.binaryproto',
                        help='resnet mean file path')
    parser.add_argument("--prototxt-path", type=str,
                        help='caffe prototxt path')
    parser.add_argument("--caffemodel-path", type=str,
                        help='caffe model path')
    return parser.parse_args()


target_height = 224
target_width = 224


def pre_process(color_img, mean, is_mirror=False):
    resized_img = cv2.resize(color_img, (target_width, target_height))
    if is_mirror:
        flip_img = cv2.flip(resized_img, 1)
        return np.transpose(flip_img, (2, 0, 1)) - mean
    else:
        return np.transpose(resized_img, (2, 0, 1)) - mean


def predict_img(net, mean, feature_layer, attr_num, img_path):
    im = cv2.imread(img_path)
    attr = np.zeros(attr_num, np.uint8)
    if im is None:
        print('%s is None' % img_path)
    else:
        resized_img = pre_process(im, mean)
        resized_img_2 = pre_process(im, mean, True)
        net.blobs['data'].reshape(2, *resized_img.shape)
        net.blobs['data'].data[0] = resized_img
        net.blobs['data'].data[1] = resized_img_2
        out = net.forward()
        score = np.mean(out[feature_layer], axis=0)
        for j in range(0, attr_num):
            if score[j] >= 0.5:
                attr[j] = 1
            else:
                attr[j] = 0
    return attr


def load_model(gpu, model_path, prototxt_path, mean_file):
    if not os.path.isfile(model_path):
        raise IOError('%s model not found.\n' % model_path)
    caffe.set_mode_gpu()
    caffe.set_device(int(gpu))
    net = caffe.Net(prototxt_path, model_path, caffe.TEST)
    proto_data = open(mean_file, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]
    print('Loaded network {:s}'.format(model_path))
    return net, mean


def main():
    args = get_arguments()
    print(args)
    gpu = args.gpu
    attr_num = int(args.attr_num)
    prototxt_path = args.prototxt_path
    caffemodel_path = args.caffemodel_path
    test_file = args.test_file
    pred_file = args.pred_file
    feature_layer = args.feature_layer
    root_folder = args.root_folder
    mean_file = args.mean_file
    # ---loading model-----
    net, mean = load_model(gpu, caffemodel_path, prototxt_path, mean_file)
    f = open(test_file, 'r')
    lines = f.readlines()
    out_f = open(pred_file, 'w')
    for idx, line in enumerate(lines):
        infos = line.strip().split()
        if idx % 100 == 0:
            print(idx)
        img_path = os.path.join(root_folder, infos[0])
        attr = predict_img(net, mean, feature_layer, attr_num, img_path)
        out_line = img_path + ' '
        for index in range(attr_num):
            out_line += str(int(attr[index])) + ' '
        out_line += '\n'
        out_f.write(out_line)


if __name__ == '__main__':
    main()