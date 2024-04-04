import sys
import time
import os
import pickle

import torch
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from mpi4py import MPI
from DataSet import MyOwnDataset
from modelV9 import Clique, DAG




def main(arg):

    name, mtype, Cq, EF, HL, layers, train_dir, act, myrank = arg

    if EF:
        Nfeat = 10
        xf = 6
        ef = 12
    else:
        Nfeat = 4
        xf = 1
        ef = 3

    seed_everything(12345)

    def train(loader):
        model.train()
        Tloss = 0
        returnData = []
        for i in range(dataset.num_classes):
            returnData.append([0, 0])
        items = 0

        for data in loader:  # Iterate in batches over the training dataset.
            # data = data.to(device)
            out = model(data)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            # print(loss.item())
            optimizer.zero_grad()  # Clear gradients.
            Tloss += float(loss) * data.y.size(0)
            items += data.y.size(0)

            pred = out.argmax(dim=1)
            for i in range(dataset.num_classes):
                corr = (pred[data.y == i] == data.y[data.y == i])
                returnData[i][0] += len(corr)
                returnData[i][1] += corr.sum().item()

        ratio = []
        for i in range(dataset.num_classes):
            if returnData[i][0] == 0:
                ratio.append(-1)
            else:
                ratio.append(returnData[i][1] / returnData[i][0])
        return float(Tloss / items), ratio

    def test(loader):
        model.eval()
        Tloss = 0
        returnData = []
        for i in range(dataset.num_classes):
            returnData.append([0, 0])
        items = 0

        for data in loader:  # Iterate in batches over the training/test dataset.
            # data = data.to(device)
            out = model(data)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            Tloss += float(loss) * data.y.size(0)
            items += data.y.size(0)

            pred = out.argmax(dim=1)
            for i in range(dataset.num_classes):
                corr = (pred[data.y == i] == data.y[data.y == i])
                returnData[i][0] += len(corr)
                returnData[i][1] += corr.sum().item()

        ratio = []
        for i in range(dataset.num_classes):
            if returnData[i][0] == 0:
                ratio.append(-1)
            else:
                ratio.append(returnData[i][1] / returnData[i][0])
        return float(Tloss / items), ratio

    N_train = 40000
    cef = True
    # N_test = 20

    dataset = MyOwnDataset(N=N_train, train_dir=train_dir, isClique=Cq)
    dataset = dataset.shuffle()
    # print(dataset[0].x, flush=True)

    if not Cq:
        UpB = 0
        DownB = 0
        for data in dataset:
            UpB = max(UpB, data.DepthUp)
            DownB = max(DownB, data.DepthDown)
        print(UpB, ' < ', len(dataset[0].edge_index_flowEdgeUp))
        print(DownB, ' < ', len(dataset[0].edge_index_flowEdgeDown), flush=True)
        if int(UpB) > len(dataset[0].edge_index_flowEdgeUp):
            raise Exception('UpB to low')

    if len(dataset) != N_train:
        print(len(dataset))
        raise Exception("Dataset has wrong size")
    # valset = dataset[N_train:]
    # dataset = dataset[:N_train]

    # if not torch.cuda.is_available():
    #     raise Exception("cuda error")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K = 5

    kfold = myrank
    if os.path.exists(name + '/' + str(kfold) + '/data.pickle'):
        return
    if not os.path.exists(name + '/' + str(kfold) + '/'):
        os.makedirs(name + '/' + str(kfold) + '/')

    if mtype in ["TransCq", "GATCq", "ResGatedCq"]:
        model = Clique(xf=xf, ef=ef, hidden_channels=HL, num_layers=layers, l_type=mtype, l_act=act, changing_edge_feature=cef)
    if mtype in ["TransLay", "SageDAG", "GatDAG"]:
        model = DAG(num_features=Nfeat, hidden_channels=HL, num_layers=layers, l_type=mtype)
    # model.to(device)
    model.reset_parameters()

    I1 = int(N_train / K * kfold)
    I2 = int(N_train / K * (kfold + 1))

    datasetTrain = ConcatDataset([dataset[:I1], dataset[I2:]])
    weight = [0, 0]
    for i in datasetTrain:
        for j in i.y:
            weight[int(j)] += 1
    print(weight)

    train_loader = DataLoader(datasetTrain, batch_size=int(N_train / K), shuffle=False)
    test_loader = DataLoader(dataset[I1:I2], batch_size=int(N_train / K), shuffle=False)

    # Rprop ASGD Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([weight[1], weight[0]], dtype=torch.float))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    x = []
    train_loss = []
    train_ratio = []
    test_loss = []
    test_ratio = []
    lr_set = []
    bestTrainLoss = 1
    bestTestLoss = 1
    epoch_start = 0
    if os.path.exists(name + '/' + str(kfold) + '/checkpoint.pickle'):
        with open(name + '/' + str(kfold) + '/checkpoint.pickle', 'rb') as f:
            [checkpoint, epoch_start] = pickle.load(f)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            with open(name + '/' + str(kfold) + '/data_sub.pickle', 'rb') as f:
                [train_loss, train_ratio, test_loss, test_ratio, x, lr_set, bestTrainLoss, bestTestLoss] = pickle.load(f)
        except:
            with open(name + '/' + str(kfold) + '/data_sub.pickle', 'rb') as f:
                [train_loss, train_ratio, test_loss, test_ratio, x, lr_set] = pickle.load(f)
                bestTrainLoss = min(train_loss)
                bestTestLoss = min(test_loss)

    for epoch in range(epoch_start + 1, 200 + 1):
        startT = time.time()
        losstrain, ratio = train(train_loader)
        train_loss.append(losstrain)
        train_ratio.append(ratio)

        losstest, ratiotest = test(test_loader)
        test_loss.append(losstest)
        test_ratio.append(ratiotest)

        if losstrain < bestTrainLoss:
            torch.save(model.state_dict(), name + '/' + str(kfold) + '/btrainmodel.pt')
            bestTrainLoss = losstrain

        if losstest < bestTestLoss:
            torch.save(model.state_dict(), name + '/' + str(kfold) + '/btestmodel.pt')
            bestTestLoss = losstest

        print('Epoch: %.1d, %.3d, %.4f' % (kfold, epoch, losstrain))
        # print(ratio)
        x.append(epoch)
        lr_set.append(scheduler._last_lr)
        scheduler.step()
        print(time.time() - startT, flush=True)
        if epoch % 10 == 0:
            with open(name + '/' + str(kfold) + '/data_sub.pickle', 'wb') as f:
                pickle.dump([train_loss, train_ratio, test_loss, test_ratio, x, lr_set, bestTrainLoss, bestTestLoss], f)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            with open(name + '/' + str(kfold) + '/checkpoint.pickle', 'wb') as f:
                pickle.dump([checkpoint, epoch], f)

    with open(name + '/' + str(kfold) + '/data.pickle', 'wb') as f:
        pickle.dump([train_loss, train_ratio, test_loss, test_ratio, x, lr_set], f)
    plt.figure(1)
    plt.plot(x, test_loss)
    plt.savefig(name + '/' + str(kfold) + '/los.png')
    plt.clf()


if __name__ == '__main__':
    file_path = 'results/'

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    if nprocs != 5:
        raise Exception(nprocs, ' processes, not 5')
    myrank = comm.Get_rank()

    EF = True
    HC = 20
    Layers = 3
    act = 'relu'

    # train_dir = '25L_15R_Combi'
    # train_dir = '20L_Combi_3'
    train_dir = str(sys.argv[1])
    HC = int(sys.argv[2])
    Layers = int(sys.argv[3])
    mtype = str(sys.argv[4])

    # ["TransCq", "GATCq", "ResGatedCq"]
    # ["TransLay", "SageDAG", "GatDAG"]

    Cq = mtype in ["TransCq", "GATCq", "ResGatedCq"]
    name = 'results/' + train_dir + '/' + mtype + '/' +'temp' + '_EF'
    name+= '_N'
    name += '_C' + str(HC) + '_L' + str(Layers)
    if act != None:
        name += act

    if not os.path.exists(name):
        try:
            os.makedirs(name)
        except:
            pass


    arg = name, mtype, Cq, EF, HC, Layers, train_dir, act, myrank
    print(arg)
    main(arg)
