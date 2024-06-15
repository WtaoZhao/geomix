import numpy as np
import math, os
from math import pi
import torch
from torch_geometric.data import Data
from scipy.spatial import distance
import copy
import scipy.sparse as sp
import uproot
import torch.nn.functional as F

def edges_to_adj(edge_source, edge_target, num_node):
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    rows, columns = adj.nonzero()
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    return adj, edge_index

def cal_Median_LeftRMS(x):
    """
    Given on 1d np array x, return the median and the left RMS
    """
    median = np.median(x)
    x_diff = x - median
    x_diffLeft = x_diff[x_diff < 0]
    rmsLeft = np.sqrt(np.sum(x_diffLeft ** 2) / x_diffLeft.shape[0])
    return median, rmsLeft

def buildConnections(eta, phi):
    """
    build the Graph based on the deltaEta and deltaPhi of input particles
    """
    phi = phi.reshape(-1, 1)
    eta = eta.reshape(-1, 1)
    dist_phi = distance.cdist(phi, phi, 'cityblock')
    indices = np.where(dist_phi > pi)
    temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
    dist_phi[indices] = dist_phi[indices] - temp
    dist_eta = distance.cdist(eta, eta, 'cityblock')
    dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)
    edge_source = np.where((dist < 0.4) & (dist != 0))[0]
    edge_target = np.where((dist < 0.4) & (dist != 0))[1]
    return edge_source, edge_target

def load_pileup(num_event, args, datadir):
    balanced = args.balanced
    edge_feature = args.edge_feature

    features = [
        'PF/PF.PT', 'PF/PF.Eta', 'PF/PF.Phi', 'PF/PF.Mass',
        'PF/PF.Charge', 'PF/PF.PdgID', 'PF/PF.IsRecoPU',
        'PF/PF.IsPU'
    ]
    tree = uproot.open(datadir)["Delphes"]
    pfcands = tree.arrays(features, entry_start=0, entry_stop=0 + num_event)

    data_list = []
    feature_events = []
    charge_events = []
    label_events = []
    edge_events = []
    nparticles = []
    nChg = []
    nNeu = []
    nChg_LV = []
    nChg_PU = []
    nNeu_LV = []
    nNeu_PU = []
    pdgid_0_events = []
    pdgid_11_events = []
    pdgid_13_events = []
    pdgid_22_events = []
    pdgid_211_events = []
    pt_events = []
    eta_events = []
    phi_events = []
    mass_events = []
    pt_avg_events = []
    eta_avg_events = []
    phi_avg_events = []

    tmp=pfcands[:]
    # print(f'len:{len(tmp)}') # number of graphs

    for i in range(num_event):
        if i % 1 == 0:
            print("processed {} events".format(i))
        event = pfcands[:][i] # type: awkward.highlevel.Record

        isCentral = (abs(event['PF/PF.Eta']) < 2.5)
        event = event[isCentral]
        pt_cut = event['PF/PF.PT'] > 0.5
        event = event[pt_cut] # select part of event
        isChg = abs(event['PF/PF.Charge']) != 0
        isChgLV = isChg & (event['PF/PF.IsPU'] == 0)
        isChgPU = isChg & (event['PF/PF.IsPU'] == 1)

        isPho = (abs(event['PF/PF.PdgID']) == 22)
        isPhoLV = isPho & (event['PF/PF.IsPU'] == 0)
        isPhoPU = isPho & (event['PF/PF.IsPU'] == 1)

        isNeuH = (abs(event['PF/PF.PdgID']) == 0)
        isNeu = abs(event['PF/PF.Charge']) == 0
        isNeuLV = isNeu & (event['PF/PF.IsPU'] == 0)
        isNeuPU = isNeu & (event['PF/PF.IsPU'] == 1)

        charge_num = np.sum(np.array(isChg).astype(int))
        neutral_num = np.sum(np.array(isNeu).astype(int))
        Chg_LV = np.sum(np.array(isChgLV).astype(int))
        Chg_PU = np.sum(np.array(isChgPU).astype(int))
        Neu_LV = np.sum(np.array(isNeuLV).astype(int))
        Neu_PU = np.sum(np.array(isNeuPU).astype(int))
        nChg.append(charge_num)
        nNeu.append(neutral_num)
        nChg_LV.append(Chg_LV)
        nChg_PU.append(Chg_PU)
        nNeu_LV.append(Neu_LV)
        nNeu_PU.append(Neu_PU)

        split_idx = int(0.7 * charge_num)
        num_particle = len(isChg)
        nparticles.append(num_particle)

        # calculate deltaR
        eta = np.array(event['PF/PF.Eta'])
        phi = np.array(event['PF/PF.Phi'])
        edge_source, edge_target = buildConnections(eta, phi)
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        edge_events.append(edge_index)
        print("done!!")

        adj, _ = edges_to_adj(edge_source, edge_target, num_particle)

        # node features
        pt = np.array(event['PF/PF.PT'])
        mass = np.array(event['PF/PF.Mass'])
        pdgid_0 = (abs(event['PF/PF.PdgID']) == 0)
        pdgid_11 = (abs(event['PF/PF.PdgID']) == 11)
        pdgid_13 = (abs(event['PF/PF.PdgID']) == 13)
        pdgid_22 = (abs(event['PF/PF.PdgID']) == 22)
        pdgid_211 = (abs(event['PF/PF.PdgID']) == 211)

        isReco = np.array(event['PF/PF.IsRecoPU'])
        chg = np.array(event['PF/PF.Charge'])
        label = np.array(event['PF/PF.IsPU'])
        pdgID = np.array(event['PF/PF.PdgID'])

        # truth label
        label = torch.from_numpy(label == 0)
        label = label.type(torch.long) # shape: (n,)

        charge_events.append(chg)
        label_events.append(label)
        name = [isChgLV, isChgPU, isNeuLV, isNeuPU]
        if i == 0:
            for node_i in range(4):
                pt_events.append(pt[name[node_i]])
                eta_events.append(eta[name[node_i]])
                phi_events.append(phi[name[node_i]])
                mass_events.append(mass[name[node_i]])
                pdgid_0_events.append([np.sum(np.array(pdgid_0 & name[node_i]).astype(int))])
                pdgid_11_events.append([np.sum(np.array(pdgid_11 & name[node_i]).astype(int))])
                pdgid_13_events.append([np.sum(np.array(pdgid_13 & name[node_i]).astype(int))])
                pdgid_22_events.append([np.sum(np.array(pdgid_22 & name[node_i]).astype(int))])
                pdgid_211_events.append([np.sum(np.array(pdgid_211 & name[node_i]).astype(int))])
                pt_avg_events.append([np.average(np.array(pt[name[node_i]]))])
                eta_avg_events.append([np.average(np.array(abs(eta[name[node_i]])))])
                phi_avg_events.append([np.average(np.array(abs(phi[name[node_i]])))])

        else:
            for node_i in range(4):
                pt_events[node_i] = np.concatenate((pt_events[node_i], pt[name[node_i]]))
                eta_events[node_i] = np.concatenate((eta_events[node_i], eta[name[node_i]]))
                phi_events[node_i] = np.concatenate((phi_events[node_i], phi[name[node_i]]))
                mass_events[node_i] = np.concatenate((mass_events[node_i], mass[name[node_i]]))
                pdgid_0_events[node_i].append(np.sum(np.array(pdgid_0 & name[node_i]).astype(int)))
                pdgid_11_events[node_i].append(np.sum(np.array(pdgid_11 & name[node_i]).astype(int)))
                pdgid_13_events[node_i].append(np.sum(np.array(pdgid_13 & name[node_i]).astype(int)))
                pdgid_22_events[node_i].append(np.sum(np.array(pdgid_22 & name[node_i]).astype(int)))
                pdgid_211_events[node_i].append(np.sum(np.array(pdgid_211 & name[node_i]).astype(int)))
                pt_avg_events[node_i].append(np.average(np.array(pt[name[node_i]])))
                eta_avg_events[node_i].append(np.average(np.array(abs(eta[name[node_i]]))))
                phi_avg_events[node_i].append(np.average(np.array(abs(phi[name[node_i]]))))

        # ffm for eta and pt
        phi = torch.from_numpy(phi)
        eta = torch.from_numpy(eta)
        pt = torch.from_numpy(pt)
        B_eta = torch.randint(0, 10, (1, 5), dtype=torch.float)
        B_pt = torch.randint(0, 10, (1, 5), dtype=torch.float)
        alpha_eta = 1
        alpha_pt = 1
        eta_ffm = (2 * pi * alpha_eta * eta).view(-1, 1) @ B_eta
        eta_ffm = torch.cat((torch.sin(eta_ffm), torch.cos(eta_ffm)), dim=1)
        pt_ffm = (2 * pi * alpha_pt * pt).view(-1, 1) @ B_pt
        pt_ffm = torch.cat((torch.sin(pt_ffm), torch.cos(pt_ffm)), dim=1)
        node_features = np.concatenate((eta_ffm, pt_ffm), axis=1)

        # one hot encoding of label
        label_copy = copy.deepcopy(label)
        label_copy[isNeu] = 2 # drop label information of neutral particles
        label_onehot = F.one_hot(label_copy)

        # one hot encoding of pdgID
        pdgID_copy = copy.deepcopy(pdgID)
        pdgID_copy[pdgid_0] = 0
        pdgID_copy[pdgid_11] = 1
        pdgID_copy[pdgid_13] = 2
        pdgID_copy[pdgid_22] = 3
        pdgID_copy[pdgid_211] = 4
        pdgID_onehot = F.one_hot(torch.from_numpy(pdgID_copy).type(torch.long))

        if edge_feature:
            node_features = np.concatenate((node_features, pdgID_onehot, label_onehot, eta.view(-1, 1), phi.view(-1, 1)), axis=1)
        else:
            node_features = np.concatenate((node_features, pdgID_onehot, label_onehot), axis=1) # raw node_feat: 20, pdgID: 5, label_onehot: 3
        node_features = torch.from_numpy(node_features)
        node_features = node_features.type(torch.float32)

        feature_events.append(node_features)
        graph = Data(x=node_features, edge_index=edge_index, y=label)
        graph.adj = adj
        graph.y_hat = label
        graph.edge_weight = torch.ones(graph.num_edges)
        graph.Charge_LV = np.where(np.array(isChgLV) == True)[0]
        graph.Charge_PU = np.where(np.array(isChgPU) == True)[0]
        if len(graph.Charge_LV) < 10:
            continue
    
        Neu_LV_indices = np.where(np.array(isNeuLV) == True)[0]
        Neu_PU_indices = np.where(np.array(isNeuPU) == True)[0]
        np.random.shuffle(Neu_LV_indices)
        np.random.shuffle(Neu_PU_indices)

        if len(Neu_LV_indices) < 2:
            continue
        num_training_LV = int(len(Neu_LV_indices) * args.labeled_ratio)
        num_training_PU = int(len(Neu_PU_indices) * args.labeled_ratio)
        num_testing_LV = len(Neu_LV_indices) - num_training_LV
        num_testing_PU = len(Neu_PU_indices) - num_training_PU

        if balanced:
            num_training_PU = num_training_LV
            num_testing_PU = num_testing_LV

        train_idx = np.concatenate((Neu_LV_indices[0:num_training_LV],
                                        Neu_PU_indices[0:num_training_PU]))
        np.random.shuffle(train_idx)

        test_idx = np.concatenate((Neu_LV_indices[num_training_LV:],
                                        Neu_PU_indices[num_training_PU:]))
        np.random.shuffle(test_idx)
        graph.labeled_index = torch.as_tensor(train_idx)
        graph.unlabeled_index = torch.as_tensor(test_idx)
        neutral_idx=np.where(np.array(isNeu))[0]
        graph.neutral_index=torch.as_tensor(neutral_idx)

        graph.num_classes = 2
        print("done!!!")
        data_list.append(graph)

    return data_list
