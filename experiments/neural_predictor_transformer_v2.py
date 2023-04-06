import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import json
import random
from argparse import ArgumentParser
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import kendalltau
import logging
import random
import sys
from collections import OrderedDict

# from architectures_loader import ArchitectureLoader
# from darts.cnn.genotypes import PRIMITIVES, PRIMITIVES_GHN
from ppuda.deepnets1m.loader import DeepNets1M
from ppuda.deepnets1m.net import Network
from ppuda.deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
import torch.multiprocessing
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

from models.transformer.encoder import Encoder
from models.transformer.batch_struct.sparse import Batch


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


class DirectedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        norm_adj = normalize_adj(adj)
        
        # print("norm_adj.shape, inputs.shape, self.weight1 = ", norm_adj.shape, inputs.shape, self.weight1.shape)
        # >>> torch.Size([10, 600, 600]) torch.Size([10, 600, 15]) torch.Size([17, 144])
        
        output1 = F.relu(torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        
        
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NeuralPredictor(nn.Module):

    # def __init__(self, initial_hidden=len(PRIMITIVES_GHN) + 2, gcn_hidden=144, gcn_layers=3, linear_hidden=128):
    # def __init__(self, initial_hidden=len(PRIMITIVES_DEEPNETS1M), gcn_hidden=144, gcn_layers=3, linear_hidden=128):
    def __init__(self, initial_hidden=len(PRIMITIVES_DEEPNETS1M), 
        n_layers=12, dim_hidden=256, dim_qk=256, dim_v=256, dim_ff=256, n_heads=16):

        super().__init__()
        # self.gcn = [DirectedGraphConvolution(initial_hidden if i == 0 else gcn_hidden, gcn_hidden)
        #             for i in range(gcn_layers)]
        # self.gcn = nn.ModuleList(self.gcn)
        # self.dropout = nn.Dropout(0.1)
        # self.fc1 = nn.Linear(gcn_hidden, linear_hidden, bias=False)
        self.fc1 = nn.Linear(initial_hidden, dim_hidden)
        self.encoder = Encoder(n_layers=n_layers, dim_in=dim_hidden, dim_out=dim_hidden, dim_hidden=dim_hidden, dim_qk=dim_qk, dim_v=dim_v, dim_ff=dim_ff, n_heads=n_heads)
        self.fc2 = nn.Linear(dim_hidden, 1, bias=False)

    def forward(self, inputs):
        numv, adj, out = inputs["num_vertices"], inputs["adjacency"], inputs["operations"]
        # numv: tensor([ 76, 186, 170, 131, 358, 211, 142, 150, 142, 321], device='cuda:0')
        # numv.shape, adj.shape, out.shape: torch.Size([10]) torch.Size([10, 600, 600]) torch.Size([10, 600, 15])
        # out = out[:,:500,:]  # to test the 600 for numv  # torch.Size([10, 500, 15])
        
        # gs = adj.size(1)  # graph node number
        # adj_with_diag = normalize_adj(adj + torch.eye(gs, device=adj.device))  # assuming diagonal is not 1
        # for layer in self.gcn:
        #     out = layer(out, adj_with_diag)  # passes into the forward of DirectedGraphConvolution
        # out = graph_pooling(out, numv)
        # out = self.fc1(out)
        # out = self.dropout(out)
        # adj[adj > 1] = 0  # temp  to be deleted for V1

        # G.values: [bsize, max(n+e), 2*dim_hidden]
        numv = [600]*len(numv)
        values = self.fc1(out)
        G = Batch(indices=None, values=values, n_nodes=numv, n_edges=None)
        _, out = self.encoder(G)
        out = self.fc2(torch.mean(out.values, dim=1)).view(-1)
        
        return out




class DeepNets1M_dataset(Dataset):
    def __init__(self, args, split=None, is_imagenet=False, virtual_edges=1, debug=False):
        self.DeepNets1M_loader = DeepNets1M.loader(split=split,  
                                 nets_dir=args.data_dir,
                                 large_images=is_imagenet,
                                 virtual_edges=virtual_edges,
                                 arch=args.arch)

        self.random_state = np.random.RandomState(0)
        self.max_nodes = 600
        self.dataset = args.dataset
        self.split = split
        self.debug = debug
        self.seed = args.seed
        self.graphs_queue = [graphs for graphs in self.DeepNets1M_loader] 
        self.val_acc_array = self.get_graph_properties()['val_acc'][self.split]
        self.val_acc_noise_array = self.get_graph_properties()['val_acc_noise'][self.split]
        self.time_array = self.get_graph_properties()['time'][self.split]
        self.converge_time_array = self.get_graph_properties()['val_acc'][self.split]


    def __len__(self):
        return len(self.DeepNets1M_loader)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0


    def get_graph_properties(self):
        with open('./data/results_%s.json' % self.dataset, 'r') as f:
            results = json.load(f)
        properties = {}
        for prop in ['val_acc', 'val_acc_noise', 'time', 'converge_time']:
            properties[prop] = {}
            for split in ['val', 'test']:
                properties[prop][split] = np.array([r[prop] for r in results[split].values()])
        
        return properties

    def get_data_from_graphbatch(self, graphs):
        data = []
        for graph in graphs:  # graphbatch has size 1 for eval
            num_vertices = graph.n_nodes
            adjacency = graph._Adj
            adjacency = F.pad(adjacency, (0, self.max_nodes - num_vertices, 0, self.max_nodes - num_vertices)).float()
            
            ops_onehot = np.zeros((self.max_nodes, len(PRIMITIVES_DEEPNETS1M)))
            i = 0
            for nodes in graph.node_feat: # for ops_onehot change the  graph.node_feat
                # for n in nodes:
                    # ops_onehot[i, PRIMITIVES_GHN[n[0]]] = 1
                ops_onehot[i, nodes[0]] = 1
                i += 1
                
            assert i == num_vertices, (i, num_vertices)
            
            operations = torch.from_numpy(ops_onehot).float(),
            operations = operations[0]  # for some reason, operation is a tuple with len of 1 instead of torch array
            mask = torch.from_numpy(np.array([i < num_vertices for i in range(self.max_nodes)], dtype=np.float32)).float(),
            mask = mask[0]
            
            data.append([num_vertices, adjacency, operations, mask])
        return data[0]  # graphbatch has size 1 for val

    def __getitem__(self, index):
        val_acc = self.val_acc_array[index].astype('float32')
        val_acc_noise = self.val_acc_noise_array[index].astype('float32')
        time = self.time_array[index].astype('float32')
        converge_time = self.converge_time_array[index].astype('float32')
        
        graphs = self.graphs_queue[index]
        num_vertices, adjacency, operations, mask = self.get_data_from_graphbatch(graphs)
        
        
        
        # print("num_vertices, adjacency, operations, mask", type(num_vertices), type(adjacency), type(operations), type(mask))
        # >>> <class 'int'> <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'>

        result = {
            "num_vertices": num_vertices, 
            "adjacency": adjacency,
            "operations": operations,
            "mask": mask,
            "val_acc": val_acc,
            "val_acc_noise": val_acc_noise,  # TODO
            "time": time,  # TODO
            "converge_time": converge_time  # TODO
        }
        # result = {
        #     "num_vertices": graph.n_nodes, # N
            # "adjacency": F.pad(graph._Adj, (0, self.max_nodes - N, 0, self.max_nodes - N)).float(), # np.pad(data[0], ((0, self.max_nodes - N),(0, self.max_nodes - N))),
        #     "operations": torch.from_numpy(ops_onehot).float(),
        #     "mask": torch.from_numpy(np.array([i < N for i in range(self.max_nodes)], dtype=np.float32)).float(),
        #     "val_acc": self.acc[index]
        #     # "test_acc": self.acc[index]
        # }
        if self.debug:
            self._check(result)
        return result


class Nb101Dataset(Dataset):
    def __init__(self, split=None, debug=False):
        self.archs_dataset = ArchitectureLoader(
            numch=[32, 32],
            batch_size=1,
            sp_cutoff=50,
            sp=False,
            image_dataset='cifar10',
            is_train=False,
            ood=None,
            root='/scratch/ssd/data/',
            verbose=False)
        self.archs_dataset.prepare_batches()
        with open('c10_results.json', 'r') as f:
            ALL_RESULTS_C10 = json.load(f)
        sgd_split = 'val_50epochs'  # 'test_1epoch'  , val_50epochs_noise
        self.acc = torch.from_numpy(np.array([ALL_RESULTS_C10['sgd'][sgd_split][str(seed)]['acc'] for seed in np.arange(1000)])).float()  # 'time', 'acc'

        with open('c10_speed_conv.json', 'r') as f:
            c10_speed_conv = json.load(f)
        conv_speed2 = np.array([int(ind[0]) / int(ind[1]) for ind in c10_speed_conv])
        print('c10_speed_conv', np.sum(conv_speed2 >= 0))
        conv_speed2[conv_speed2 < 0] = 2
        self.acc = torch.from_numpy(conv_speed2).float()

        # self.acc = np.array([ALL_RESULTS_C10['sgd'][sgd_split][seed]['acc'] for seed in np.arange(1000)])
        self.random_state = np.random.RandomState(0)
        self.sample_range = np.arange(500) if split == 'train' else np.arange(500,1000)
        self.max_nodes = 600
        self.split = split
        self.debug = debug
        self.seed = 0

    def __len__(self):
        return len(self.sample_range)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0


    def __getitem__(self, index):
        index = self.sample_range[index]
        data = self.archs_dataset[index]
        N = data[0].shape[0]
        ops_onehot = np.zeros((self.max_nodes, len(PRIMITIVES_GHN)+2))
        i = 0
        for nodes in data[1]:
            for n in nodes:
                ops_onehot[i, PRIMITIVES_GHN[n[0]]] = 1
                i += 1
        assert i == N, (i, N)

        result = {
            "num_vertices": N,
            "adjacency": F.pad(data[0], (0, self.max_nodes - N, 0, self.max_nodes - N)).float(), # np.pad(data[0], ((0, self.max_nodes - N),(0, self.max_nodes - N))),
            "operations": torch.from_numpy(ops_onehot).float(),
            "mask": torch.from_numpy(np.array([i < N for i in range(self.max_nodes)], dtype=np.float32)).float(),
            "val_acc": self.acc[index]
            # "test_acc": self.acc[index]
        }
        if self.debug:
            self._check(result)
        return result

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, tuple):
        return tuple(to_cuda(t) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t) for t in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def get_logger():
    time_format = "%m/%d %H:%M:%S"
    fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
    formatter = logging.Formatter(fmt, time_format)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def accuracy_mse(predict, target, scale=100.):
    predict = predict.detach()
    target = target
    return F.mse_loss(predict, target)


def main():
    # valid_splits = ["172", "334", "860", "91-172", "91-334", "91-860", "denoise-91", "denoise-80", "all"]
    parser = ArgumentParser()
    parser.add_argument("--gcn_hidden", type=int, default=256) # originally 144
    parser.add_argument("--seed", type=int, default=222)  # originally 222 for v1
    parser.add_argument("--train_batch_size", default=40, type=int)
    parser.add_argument("--eval_batch_size", default=100, type=int)  # original 1000
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--wd", "--weight_decay", default=1e-3, type=float)
    parser.add_argument("--train_print_freq", default=None, type=int)
    parser.add_argument("--eval_print_freq", default=10, type=int)
    parser.add_argument("--visualize", default=False, action="store_true")
    
    # argument for DeepNet1M:
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='image dataset: cifar10/imagenet/PennFudanPed.')
    parser.add_argument('-D', '--data_dir', type=str, default='./data',
                    help='where image dataset and DeepNets-1M are stored')
    
    mode = 'eval'
    is_train_net = mode == 'train_net'
    is_eval = mode == 'eval'
    parser.add_argument('--arch', type=str,
                            default='DARTS' if is_train_net else None,
                            help='one of the architectures: '
                                 'string for the predefined genotypes such as DARTS; '
                                 'the architecture index from DeepNets-1M')
        
    args = parser.parse_args()

    


    reset_seed(args.seed)
    # print("********* torch.seed={}, numpy.seed={}, args.seed={} ********".format(torch.seed(), np.random.get_state()[1][0], args.seed))
    
    target_property = 'val_acc'  # one of ['val_acc', 'val_acc_noise', 'time', 'converge_time']
    is_imagenet = args.dataset == 'imagenet'
    virtual_edges = 50  # default values
    dataset = DeepNets1M_dataset(args, split='val', is_imagenet=is_imagenet, virtual_edges=virtual_edges)
    dataset_test = DeepNets1M_dataset(args, split='test', is_imagenet=is_imagenet, virtual_edges=virtual_edges)
    
    # dataset = Nb101Dataset(split='train')
    # dataset_test = Nb101Dataset(split='val')
    
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(dataset_test, batch_size=args.eval_batch_size)

    net = NeuralPredictor(dim_hidden=args.gcn_hidden)
    net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    logger = get_logger()

    net.train()
    print("Training for Target_property = ", target_property)
    for epoch in tqdm(range(args.epochs), position=0, leave=True):
        meters = AverageMeterGroup()
        lr = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(data_loader):
            batch = to_cuda(batch)
            # target = batch["val_acc"]
            target = batch[target_property]
            predict = net(batch)
            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            mse = accuracy_mse(predict, target)
            meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))
            if (args.train_print_freq and step % args.train_print_freq == 0) or \
                    step + 1 == len(data_loader):
                logger.info("Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s",
                            epoch + 1, args.epochs, step + 1, len(data_loader), lr, meters)
        lr_scheduler.step()

    net.eval()
    meters = AverageMeterGroup()
    predict_, target_ = [], []
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch = to_cuda(batch)
            # target = batch["val_acc"]
            target = batch[target_property]
            predict = net(batch)
            predict_.append(predict.cpu().numpy())
            target_.append(target.cpu().numpy())
            meters.update({"loss": criterion(predict, target).item(),
                        "mse": accuracy_mse(predict, target).item()}, n=target.size(0))

            if (args.eval_print_freq and step % args.eval_print_freq == 0) or \
                    step % 10 == 0 or step + 1 == len(test_data_loader):
                logger.info("Evaluation Step [%d/%d]  %s", step + 1, len(test_data_loader), meters)
    predict_ = np.concatenate(predict_)
    target_ = np.concatenate(target_)
    print('%d/%d samples' % (len(predict_), len(target_)))
    logger.info("Kendalltau: %.6f", kendalltau(predict_, target_)[0])
    print("Training finished for Target_property = ", target_property)


if __name__ == "__main__":
    main()