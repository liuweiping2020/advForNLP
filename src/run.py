# coding: UTF-8
import time
import torch
import numpy as np
from trains.train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--epsilon', default=8 / 50, type=int)
parser.add_argument('--alpha', default=2, type=int, help='Step size')
parser.add_argument('--delta-init', default='zero', choices=['zero', 'random'],
                    help='Perturbation initialization method')
parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
parser.add_argument('--delta_init', default='random', choices=['zero', 'random'],
                    help='Perturbation initialization method')
parser.add_argument('--sgdflag', default="fgsm", type=str, help='choose a model: pgd,free,fgsm')
parser.add_argument('--minibatch_replays', default=8, type=int)

args = parser.parse_args()


def doTrain():
    dataset = 'THUCNews'  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from commons.utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    config.epsilon = args.epsilon
    config.alpha = args.alpha
    config.attack_iters = args.attack_iters
    config.sgdflag = args.sgdflag
    config.delta_init = args.delta_init
    config.minibatch_replays = args.minibatch_replays
    train(config, model, train_iter, dev_iter, test_iter)


if __name__ == '__main__':
    doTrain()

    pass
