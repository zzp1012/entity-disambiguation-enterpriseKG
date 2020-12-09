# import some usual packages
import logging
import argparse
import os
import pathlib

# import machine learing or computing libraries
import pandas as pd 
import numpy as np
import networkx as nx 
from node2vec import Node2Vec
import dgl
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as fn
import itertools

# the path for pretrained node embedding vectors.
pretrain_path = "./pretrained"
# the path for data.
data_path = "../data"

# not user-changable hyperparameters settings.
NUM_OF_WORKERS = 10
NUM_OF_WINDOWS = 10
BATCH_WORDS = 4


# set the logger
logging.basicConfig(
                    # filename = "logfile",
                    # filemode = "w+",
                    format='%(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger("NED")

# logistic regression model
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# fully connected layer
class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# graph convolution network
class GCN(nn.Module):
    def __init__(self, embed_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(embed_size, embed_size)
        self.conv2 = GraphConv(embed_size, embed_size)
    
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = torch.relu(h)
        h = self.conv2(graph, h)
        return h


def precision(y_pred, y_truth):
    """
    y_pred: the predicted value.
    y_truth: the ground truth value.
    return: precision.
    """
    return np.sum(y_pred * y_truth) / np.sum(y_pred)


def recall(y_pred, y_truth):
    """
    y_pred: the predicted value.
    y_truth: the ground truth value.
    return: recall.
    """
    return np.sum(y_pred * y_truth) / np.sum(y_truth)


def always_true(dataset):
    """
    dataset: the whole dataset including train and test.
    effect: evaluate the baseline method in terms of recall and precision.
    """
    # once the a.name == b.name, then we consider it as true.
    y_pred = np.ones(len(dataset))
    y_truth = dataset[:, -1]
    # show the precision and recall
    logger.info("always_true's precision {0:.3f}".format(precision(y_pred, y_truth)))
    logger.info("always_true's recall {0:.3f}".format(recall(y_pred, y_truth)))


def scca(graph, dataset):
    """
    graph: the undirect graph.
    dataset: the whole dataset including train and test.
    effect: evaluate the baseline method in terms of recall and precision.
    """
    # Although the actual graph is undirected, we can not convert it to undirected type.
    graph = graph.to_networkx()
    # find all the components
    components = list()
    for comp in list(nx.strongly_connected_components(graph)):
        if len(comp) > 1:
            components.append(comp)
    # define a function for check whether u and v are belong to the same group.
    def is_same_group(u, v):
        for comp in components:
            if u in comp and v in comp:
                return True
        return False
    # once they are belong to one component, we will consider them as the same entity.
    y_pred = np.array(list(map(is_same_group, dataset[:,0], dataset[:, 1])))
    y_truth = dataset[:, -1]
    # show the precision and recall
    logger.info("scca's precision {0:.3f}".format(precision(y_pred, y_truth)))
    logger.info("scca's recall {0:.3f}".format(recall(y_pred, y_truth)))


def shallow_embedding(graph, train, test, args):
    """
    graph: the undirect graph.
    train: the training dataset.
    test: the testing dataset.
    args: dict containing all the program arguments.
    effect: evaluate the node2vec method in terms of recall and precision.
    """
    # Although the actual graph is undirected, we can not convert it to undirected type.
    graph = graph.to_networkx()
    pretrained = pathlib.Path(os.path.join(pretrain_path, "node2vec-embedd.txt"))
    if not pretrained.is_file() or args.fit:
        logger.info("There is no pretrained node2vec embedding vectors.\n\t Plz wait for a moment for us to obtain the new embedding vectors.")
        if not pathlib.Path(pretrain_path).is_dir():
            logger.info("There is no " + pretrain_path + ". Therefore, we make one")
            os.mkdir(pretrain_path)
        node2vec = Node2Vec(graph, 
                            dimensions=args.embed, 
                            walk_length=args.walk_len, 
                            num_walks=args.num_walks, 
                            workers=NUM_OF_WORKERS)
        node2vec_fit = node2vec.fit(window=NUM_OF_WINDOWS, 
                                    min_count=1,
                                    batch_words=BATCH_WORDS)
        node2vec_fit.wv.save_word2vec_format(os.path.join(pretrain_path, "node2vec-embedd.txt"))
    else: 
        logger.info("We have already pretrained node embedding vectors, therefore, we turn to use them.")
    
    # read the pretrained node embedding vectors.
    try:
        embeddings = pd.read_table(os.path.join(pretrain_path, "node2vec-embedd.txt"), sep = " ", header = None)
    except:
        logger.error("Data load error. Plz check your data_path")
        os._exit(0)
    
    # create real training set and test set
    def pair2vec(u, v):
        u_vec = embeddings.iloc[int(u), 1:].values
        v_vec = embeddings.iloc[int(v), 1:].values
        return np.abs(u_vec - v_vec)
    train_x = torch.from_numpy(np.array(list(map(pair2vec, train[:, 0], train[:, 1]))))
    train_y = torch.from_numpy(train[:,-1])
    test_x = torch.from_numpy(np.array(list(map(pair2vec, test[:, 0], test[:, 1]))))
    
    # create the fully connected nn.
    fc = FC(args.embed, args.hidden)
    logger.info(fc)
    
    # train the network
    optimizer = torch.optim.Adam(fc.parameters(), lr=args.lr)
    for epoch in range(args.max_itr):
        fc.train()
        logits = fc(train_x.float())
        loss = nn.BCELoss()(logits.squeeze(dim = -1), train_y.float().squeeze(dim = -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.debug('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    # evaluate the node2vec method.
    fc.eval()
    with torch.no_grad():
        logits = fc(test_x.float())
        y_pred = (logits > 0.5).float().squeeze(dim = -1).numpy()
        y_truth = test[:, -1].ravel()
        # show the precision and recall
        logger.info("node2vec's precision {0:.3f}".format(precision(y_pred, y_truth)))
        logger.info("node2vec's recall {0:.3f}".format(recall(y_pred, y_truth)))



def gcn(graph, train, test, args):
    """
    graph: the undirect graph.
    train: the training dataset.
    test: the testing dataset.
    args: dict containing all the program arguments.
    effect: evaluate the gcn method in terms of recall and precision.
    """
    # defin the learnable embedding. 
    embed = nn.Embedding(graph.number_of_nodes(), args.embed)

    # define the model.
    gcn_model = GCN(args.embed)
    fc_model = FC(args.embed, args.hidden)
    if args.area:
        lr_model = LogisticRegression()
    logger.info(gcn_model)
    logger.info(fc_model)
    if args.area:
        logger.info(lr_model)

    # starts training
    if args.area:
        optimizer = torch.optim.Adam(itertools.chain(lr_model.parameters(), fc_model.parameters(), gcn_model.parameters(), embed.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(itertools.chain(fc_model.parameters(), gcn_model.parameters(), embed.parameters()), lr=args.lr)
    for epoch in range(args.max_itr * 4):
        inputs = embed.weight

        # compute the embedding vectors
        embeds = gcn_model(graph, inputs)
        
        # define the function which will be used later for creating real dataset
        def pair2vec(u, v):
            u_vec = embeds[int(u), 0:]
            v_vec = embeds[int(v), 0:]
            return torch.abs(u_vec - v_vec).detach().numpy()

        # create real training set and test set    

        train_x = torch.from_numpy(np.array(list(map(pair2vec, train[:, 0], train[:, 1])))).float()
        train_y = torch.from_numpy(train[:,-1]).float()
        logits = fc_model(train_x)

        if args.area:
            train_area = torch.from_numpy(train[:,2]).float()
            train_area = torch.reshape(train_area, (-1,1))
            logits = lr_model(torch.cat((logits, train_area), 1))

        loss = nn.BCELoss()(logits.squeeze(dim = -1), train_y.squeeze(dim = -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.debug('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

        if epoch == args.max_itr - 1:
            with torch.no_grad():
                test_x = torch.from_numpy(np.array(list(map(pair2vec, test[:, 0], test[:, 1])))).float()
                logits = fc_model(test_x)

                if args.area:
                    test_area = torch.from_numpy(test[:,2]).float()
                    test_area = torch.reshape(test_area, (-1, 1))
                    logits = lr_model(torch.cat((logits, test_area), 1))

                y_pred = (logits >= 0.5).float().squeeze(dim = -1).numpy()
                y_truth = test[:, -1].ravel()
                # show the precision and recall
                logger.info("gcn's precision {0:.3f}".format(precision(y_pred, y_truth)))
                logger.info("gcn's recall {0:.3f}".format(recall(y_pred, y_truth)))


def data_splitting(split_ratio):
    """
    split_ratio: training data / total data 
    return: original dataset, train set and test set.
    """
    # read the dataset from labels.csv
    try:
        dataset = pd.read_csv(os.path.join(data_path, "labels.csv")).values
    except:
        logger.error("Data load error. Plz check your data_path")
        os._exit(0)
    # shuffle the data
    np.random.shuffle(dataset)
    
    # check split_ratio
    if split_ratio < 0 or len(dataset) - round(len(dataset) * split_ratio) < 5:
        logger.warning("The splitting ratio is illegal. Turn to Default 0.7")
        split_ratio = 0.7
    
    # create train and test
    train = dataset[0:round(len(dataset) * split_ratio)]
    test = dataset[round(len(dataset) * split_ratio):]
    return dataset, train, test


def build_graph():
    """
    return: dgl.graph, an enterprise KG built from data.
    """
    try:
        com2com = pd.read_csv(os.path.join(data_path, "com2com.csv"))
        per2com = pd.read_csv(os.path.join(data_path, "person2com.csv"))
    except:
        logger.error("Data load error. Plz check your data_path")
        os._exit(0)
    
    # create src list and dst list
    src = np.concatenate([com2com.iloc[:,2].values, per2com.iloc[:,2].values]) 
    dst = np.concatenate([com2com.iloc[:,3].values, per2com.iloc[:,3].values])
    
    # create undirected graph
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    return dgl.graph((u, v))


def args_parser():
    """
    return: args containing programs arguments.
    """
    parser = argparse.ArgumentParser(description='Named Entity Disambiguation')

    # set the random seed
    parser.add_argument("--seed", type = int, default = 1, 
                        help= "the random seed. Default 1.")

    # set the data splitting ratio
    parser.add_argument("--split_ratio", type = float, default = 0.7, 
                        help= "the data splitting ratio. Default 0.7.")

    # set the length of random walk for node2vec
    parser.add_argument("--walk_len", type = int, default = 30, 
                        help= "the length of random walk for node2vec. Default 30.")

    # set the number of walks for node2vec
    parser.add_argument("--num_walks", type = int, default = 200, 
                        help= "the number of walks for node2vec. Default 200.")

    # set the size of hidden layer of fully connected layer
    parser.add_argument("--hidden", type = int, default = 8, 
                        help= "the size of hidden layer of fully connected layer. Default 8.")

    # set the maximum iteration
    parser.add_argument("--max_itr", type = int, default = 50, 
                        help= "the maximum nunmber of iterations. Default 50.")

    # set the size of node embedding vector
    parser.add_argument("--embed", type = int, default = 10, 
                        help= "the size of node embedding vector. Default 10.")

    # set the learning rate of GCN
    parser.add_argument("--lr", type = float, default = 0.01, 
                        help= "the learning rate of GCN. Default 0.01.")

    # set if fit a new node2vec model
    parser.add_argument("-f", "--fit", action= "store_true", dest= "fit", 
                        help= "enable fit a new node2vec model.")

    # set if using debug mod
    parser.add_argument("-a", "--area", action= "store_true", dest= "area", 
                        help= "enable area and logistic regression to improve the model")

    # set if using debug mod
    parser.add_argument("-v", "--verbose", action= "store_true", dest= "verbose", 
                        help= "enable debug info output.")

    args = parser.parse_args()

    # check if the walk length is illegal.
    if args.walk_len < 5 or args.walk_len > 1000:
        logger.warning("The walk length for running node2vec is illegal. Turn to default 30.")
        args.walk_len = 30
    
    # check if the number of walks is illegal.
    if args.num_walks < 20 or args.num_walks > 1000:
        logger.warning("The number of walks for running node2vec is illegal. Turn to default 200.")
        args.num_walks = 200

    # check if the size of embedding vector is illegal.
    if args.embed < 2 or args.embed > 100:
        logger.warning("The size of embedding vector is illegal. Turn to default 10.")
        args.embed = 10

    # check if the walk length is illegal
    if args.hidden > args.embed or args.hidden < 1:
        logger.warning("The size of hidden layer of full connected layer is illegal. Turn to default round(0.8 * args.embed).")
        args.hidden = round(0.8 * args.embed)

    return args


def main():
    # get program arguments 
    args = args_parser()

    # set the logger
    logger.setLevel(logging.DEBUG)
    if not args.verbose:
        logger.setLevel(logging.INFO)
    logger.debug("--------DEBUG enviroment start---------")

     # show the hyperparameters
    logger.info("---------hyperparameter setting---------")
    logger.info(args)

    # set the random seed.
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed) 

    # build the graph from edge lists.
    logger.info("------------Construct graph-------------")
    graph = build_graph()
    logger.info('Enterprise KG have %d nodes.' % graph.number_of_nodes())
    logger.info('Enterprise KG have %d edges.' % graph.number_of_edges())

    # data splitting
    logger.info("-------------Data splitting-------------")
    dataset, train, test = data_splitting(args.split_ratio)

    # baseline method
    logger.info("-----------Baseline evaluation----------")
    # first baseline method: always true
    always_true(dataset)
    # second baseline method: scca
    scca(graph, dataset)

    # shallow embedding
    logger.info("-----------node2vec evaluation----------")
    shallow_embedding(graph, train, test, args)

    # gcn
    logger.info("-------------GCN evaluation-------------")
    gcn(graph, train, test, args)

    logger.info("-------------End evaluation-------------")


if __name__ == "__main__":
    main()
