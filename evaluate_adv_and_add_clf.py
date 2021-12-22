import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import cluster
import train_func as tf
import utils

manualSeed = 100
torch.manual_seed(manualSeed)  # pytorch random seed
np.random.seed(manualSeed)  # numpy random seed
torch.backends.cudnn.deterministic = True

def svm(args, train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test


def knn(args, train_features, train_labels, test_features, test_labels):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.

    Options:
        k (int): top k features for kNN
    
    """
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=args.k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = utils.compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc

def nearsub_each_class(args, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1 # should be correct most of the time
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(), 
                                          num_classes=num_classes, stack=False)

    print(features_sort[0][:10])
    print(features_sort[1][:10])

    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=args.n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)
        print(score_pca_j, score_pca_j.shape)

        svd = TruncatedSVD(n_components=args.n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)

    for j in range(num_classes):
        acc_pca = utils.compute_accuracy(test_predict_pca[test_labels==j], test_labels[test_labels==j].numpy())
        acc_svd = utils.compute_accuracy(test_predict_svd[test_labels==j], test_labels[test_labels==j].numpy())
        print('Class {}'.format(j))
        print('PCA: {}'.format(acc_pca))
        print('SVD: {}'.format(acc_svd))
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))


def nearsub(args, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1 # should be correct most of the time
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(), 
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=args.n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=args.n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_svd

def kmeans(args, train_features, train_labels):
    """Perform KMeans clustering. 
    
    Options:
        n (int): number of clusters used in KMeans.

    """
    return cluster.kmeans(args, train_features, train_labels)

def ensc(args, train_features, train_labels):
    """Perform Elastic Net Subspace Clustering.
    
    Options:
        gam (float): gamma parameter in EnSC
        tau (float): tau parameter in EnSC

    """
    return cluster.ensc(args, train_features, train_labels)

def get_data(trainloader, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (batch_imgs, batch_lbls) in enumerate(train_bar):
        features.append(batch_imgs.view(-1,len(batch_imgs)).cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)

def pred_acc(model, dataloader):
    correct = 0 
    total = 0
    for data, label in dataloader:
        pred = model(data.to(DEVICE))
        correct += (pred.argmax(dim=1) == label.to(DEVICE)).sum()
        total += len(label)
    print('Pred Acc:{:.4f}'.format(correct/total))

def train_clf_and_adv_attack(net, trainloader, testloader): 
    clf = nn.Linear(128, 10).to(DEVICE)
    optimizer = optim.Adam(list(clf.parameters()) + list(net.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    train_epoch = 10
    for epoch in range(train_epoch):
        correct = 0
        total = 0
        net.train()
        clf.train()
        for i, (data, label) in enumerate(trainloader):
            feat = net(data.to(DEVICE))
            pred = clf(feat)

            loss = criterion(pred, label.long().to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (torch.argmax(pred, dim=1) == label.to(DEVICE)).sum()
            total += len(label)
        print(epoch, 'train acc:', correct.float()/total)

        correct = 0
        total = 0
        net.eval()
        clf.eval()
        for i, (data, label) in enumerate(testloader):
            feat = net(data.to(DEVICE))
            pred = clf(feat)

            correct += (torch.argmax(pred, dim=1) == label.to(DEVICE)).sum()
            total += len(label)
        print(epoch, 'test acc:', correct.float()/total)

    # Discriminator
    class new_model(nn.Module):
        def __init__(self, base, cls):
            super(new_model, self).__init__()
            self.base = base
            self.clf = clf

        def forward(self, input):
            return self.clf(self.base(input))
    
    newmodel = new_model(net, clf)
    epsilon = 0.3
    pgd_iter = 10
    pgd_iter_step = 0.04
    def eval_adv_test_whitebox(model, device, test_loader, attacker, early_stop=False):
        model.eval()
        test_loss=0
        correct=0
        nat_accu = 0
        test_number=0
        count=0
        for data, target in test_loader:
            count += 1
            if early_stop and count > 2:break
            data, target = data.to(device), target.to(device)
            adv_data = attacker.perturb(data, target)
            delta = torch.clamp((adv_data - data).detach(), -epsilon, epsilon)
            #delta = torch.clamp((adv_data - data).detach(), -1.0, 1.0)
            adv_data = data.detach() + delta
            output = model(adv_data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred = model(adv_data).argmax(dim=1).cpu()
            #print('Success: {:.2f}%'.format((pred != target.cpu()).float().mean().item() * 100))
            nat = model(data)
            nat_pred = nat.max(1, keepdim=True)[1]
            nat_accu += nat_pred.eq(target.view_as(nat_pred)).sum().item()
            test_number += data.size(0)

        test_loss /= test_number
        nat_accuracy = 100.0 * nat_accu / test_number
        print('Test: Average loss: {:.4f}, nat_accuracy: {:.2f}%, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, nat_accuracy, correct, test_number,
            100. * correct / test_number))
        
        return adv_data

    from advertorch.attacks import GradientSignAttack, LinfPGDAttack, LBFGSAttack, MomentumIterativeAttack, CarliniWagnerL2Attack
    
    print('Running FGSM attack')
    attacker = GradientSignAttack(
                newmodel, loss_fn=nn.CrossEntropyLoss(), eps=epsilon,
                clip_min=0.0, clip_max=1.0, targeted=False)
    fgsm_adv_data = eval_adv_test_whitebox(newmodel, DEVICE, testloader, attacker)    

    print('Running LinfPGD attack')
    attacker = LinfPGDAttack(
                newmodel, loss_fn=nn.CrossEntropyLoss(), eps=epsilon,
                nb_iter=pgd_iter, eps_iter=pgd_iter_step, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)
    pgd_adv_data = eval_adv_test_whitebox(newmodel, DEVICE, testloader, attacker)

def adv_attack(net, testloader): 
    epsilon = 0.3
    pgd_iter = 10
    pgd_iter_step = 0.04
    def eval_adv_test_whitebox(model, device, test_loader, attacker, early_stop=False):
        model.eval()
        test_loss=0
        correct=0
        nat_accu = 0
        test_number=0
        count=0
        for data, target in test_loader:
            count += 1
            if early_stop and count > 2:break
            data, target = data.to(device), target.to(device)
            adv_data = attacker.perturb(data, target)
            delta = torch.clamp((adv_data - data).detach(), -epsilon, epsilon)
            #delta = torch.clamp((adv_data - data).detach(), -1.0, 1.0)
            adv_data = data.detach() + delta
            output = model(adv_data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred = model(adv_data).argmax(dim=1).cpu()
            #print('Success: {:.2f}%'.format((pred != target.cpu()).float().mean().item() * 100))
            nat = model(data)
            nat_pred = nat.max(1, keepdim=True)[1]
            nat_accu += nat_pred.eq(target.view_as(nat_pred)).sum().item()
            test_number += data.size(0)

        test_loss /= test_number
        nat_accuracy = 100.0 * nat_accu / test_number
        print('Test: Average loss: {:.4f}, nat_accuracy: {:.2f}%, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, nat_accuracy, correct, test_number,
            100. * correct / test_number))
        
        return adv_data

    from advertorch.attacks import GradientSignAttack, LinfPGDAttack, LBFGSAttack, MomentumIterativeAttack, CarliniWagnerL2Attack
    
    print('Running FGSM attack')
    attacker = GradientSignAttack(
                net, loss_fn=nn.CrossEntropyLoss(), eps=epsilon,
                clip_min=0.0, clip_max=1.0, targeted=False)
    fgsm_adv_data = eval_adv_test_whitebox(net, DEVICE, testloader, attacker)    

    print('Running LinfPGD attack')
    attacker = LinfPGDAttack(
                net, loss_fn=nn.CrossEntropyLoss(), eps=epsilon,
                nb_iter=pgd_iter, eps_iter=pgd_iter_step, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)
    pgd_adv_data = eval_adv_test_whitebox(net, DEVICE, testloader, attacker)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
    parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
    parser.add_argument('--nearsub', help='evaluate using Nearest Subspace', action='store_true')
    parser.add_argument('--kmeans', help='evaluate using KMeans', action='store_true')
    parser.add_argument('--ensc', help='evaluate using Elastic Net Subspace Clustering', action='store_true')
    parser.add_argument('--pred', help='Directly predict', action='store_true')
    parser.add_argument('--adv_attack', help='Use adv attack', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    

    parser.add_argument('--k', type=int, default=5, help='top k components for kNN')
    parser.add_argument('--n', type=int, default=10, help='number of clusters for cluster (default: 10)')
    parser.add_argument('--gam', type=int, default=300, 
                        help='gamma paramter for subspace clustering (default: 100)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='tau paramter for subspace clustering (default: 1.0)')
    parser.add_argument('--n_comp', type=int, default=30, help='number of components for PCA (default: 30)')
    parser.add_argument('--save', action='store_true', help='save labels')
    parser.add_argument('--data_dir', default='./data/', help='path to dataset')
    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## load model
    params = utils.load_params(args.model_dir)
    net, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True)
    net = net.to(DEVICE).eval()

    # params = utils.load_params(args.model_dir)
    # net_base, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True)
    # net_base = net_base.to(DEVICE).eval()
    
    # clf_ckpt_path = os.path.join(args.model_dir, 'checkpoints', 'model-clf-epoch{}.pt'.format(args.epoch))
    # clf = nn.Linear(128, 10).to(DEVICE).eval()
    # clf.load_state_dict(torch.load(clf_ckpt_path))

    # class net_full(nn.Module):
    #     def __init__(self,base_model, clf):
    #         super(net_full, self).__init__()
    #         self.base_model = base_model
    #         self.clf = clf

    #     def predict(self, feat):
    #         pred = self.clf(feat)
    #         return pred

    #     def get_feat(self, x):
    #         return self.base_model(x)

    #     def forward(self, x):
    #         feat = self.base_model(x)
    #         pred = self.clf(feat)
    #         return pred
    
    # net = net_full(net_base, clf)
    
    # get train features and labels
    train_transforms = tf.load_transforms('test')
    trainset = tf.load_trainset(params['data'], train_transforms, train=True, path=args.data_dir)
    if 'lcr' in params.keys(): # supervised corruption case
        trainset = tf.corrupt_labels(params['corrupt'])(trainset, params['lcr'], params['lcs'])
    new_labels = trainset.targets
    trainloader = DataLoader(trainset, batch_size=200)
    train_features, train_labels = tf.get_features(net, trainloader)
    # train_features, train_labels = get_data(trainloader)

    # get test features and labels
    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False)
    testloader = DataLoader(testset, batch_size=200)
    test_features, test_labels = tf.get_features(net, testloader)
    # test_features, test_labels = get_data(testloader)

    if args.svm:
        svm(args, train_features, train_labels, test_features, test_labels)
    if args.knn:
        knn(args, train_features, train_labels, test_features, test_labels)
    if args.nearsub:
        nearsub_each_class(args, train_features, train_labels, test_features, test_labels)
    if args.kmeans:
        kmeans(args, train_features, train_labels)
    if args.ensc:
        ensc(args, train_features, train_labels)
    if args.pred:
        pred_acc(net, testloader)
    if args.adv_attack:
        # adv_attack(net, testloader)
        train_clf_and_adv_attack(net, trainloader, testloader)
