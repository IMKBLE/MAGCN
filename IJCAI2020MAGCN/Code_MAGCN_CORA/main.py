import argparse
import os
import tensorflow as tf
from utils.classifier import Classifier
from Network.Trainer import Trainer
from utils import process
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.fftpack import fft


def parse_args():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser(description="Run gate.")
    parser.add_argument('--dataset', nargs='?', default='cora', help='Input dataset')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate. Default is 0.001.')
    parser.add_argument('--dlr', type=float, default=3e-5, help='D Learning rate. Default is 0.001.')
    parser.add_argument('--n-epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--hidden-dims1', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--hidden-dims2', type=list, nargs='+', default=[512, 512], help='Number of dimensions2.')
    parser.add_argument('--lambda-', default=0.5, type=float, help='Parameter controlling the contribution of edge '
                                                                 'reconstruction in the loss function.')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout.')
    parser.add_argument('--gradient_clipping', default=5.0, type=float, help='gradient clipping')
    return parser.parse_args()


def main(args):
    """
    Pipeline for Graph Attention Auto-encoder.
    G, X, Y, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    print('Graph的维度：' + str(G.shape))
    print('Content的维度：' + str(X.shape))
    Label = np.array([np.argmax(l) for l in Y])
    print('Label的维度：' + str(Label.shape))
    # add feature dimension size to the beginning of hidden_dims
    feature_dim = X.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims
    print('隐层单元的维度：' + str(args.hidden_dims))
    # prepare the data
    """

    # data_z = sio.loadmat('database/HW/nhandwritten_2views.mat')
    # data_dict = dict(data_z)
    # X1 = data_dict['x1']
    # X2 = data_dict['x2']
    # Label = data_dict['gt'].T
    # Label = np.squeeze(np.array(Label))
    # data_G = sio.loadmat('database/HW/hw5.mat')
    # g = dict(data_G)['hw5']
    # G = sp.coo_matrix(dict(data_G)['hw5'])
    # G_tf, S, R = process.prepare_graph_data(G)
    #
    # feature_dim1 = X1.shape[1]
    # args.hidden_dims1 = [feature_dim1] + args.hidden_dims1
    # feature_dim2 = X2.shape[1]
    # args.hidden_dims2 = [feature_dim2] + args.hidden_dims2
    #
    # print('Graph的维度：' + str(G.shape))
    # print('Content1的维度：' + str(X1.shape))
    # print('Content2的维度：' + str(X2.shape))
    # print('Label的维度：' + str(Label.shape))
    # print('隐层单元1的维度：' + str(args.hidden_dims1))
    # print('隐层单元2的维度：' + str(args.hidden_dims2))
    #
    # # PreTrain the Model
    # # fin = False
    # trainer = Trainer(args)
    # _ = trainer.assign(G_tf, X1, S, R, G_tf, X2, S, R)
    # # trainer(G_tf, X, S, R, Label, fin)
    # # Fintune the Model
    # fin = True
    # trainer(G_tf, X1, S, R, G_tf, X2, S, R, Label, fin)
    """
    Pipeline for Graph Attention Auto-encoder.
    """
    G, X, Y, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    print('Graph的维度：' + str(G.shape))
    print('Content的维度：' + str(X.shape))
    Label = np.array([np.argmax(l) for l in Y])
    print('Label的维度：' + str(Label.shape))
    # add feature dimension size to the beginning of hidden_dims
    feature_dim1 = X.shape[1]
    args.hidden_dims1 = [feature_dim1] + args.hidden_dims1
    X2 = fft(X)
    feature_dim2 = X2.shape[1]
    args.hidden_dims2 = [feature_dim2] + args.hidden_dims2

    print('隐层单元1的维度：' + str(args.hidden_dims1))
    print('隐层单元2的维度：' + str(args.hidden_dims2))
    # prepare the data
    G_tf, S, R = process.prepare_graph_data(G)
    # PreTrain the Model
    # fin = False
    trainer = Trainer(args)
    _ = trainer.assign(G_tf, X, S, R, G_tf, X2, S, R)
    # trainer(G_tf, X, S, R, Label, fin)
    # Fintune the Model
    fin = True
    trainer(G_tf, X, S, R, G_tf, X2, S, R, Label, fin)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main(args)
