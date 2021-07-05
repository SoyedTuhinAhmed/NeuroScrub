"""
Author: Soyed Tuhin Ahmed
"""
import torch
import torchvision as tv
import numpy as np
import os
from bin_model import *
from retention_faults_sim import *
import argparse
from torchvision.datasets import ImageFolder


def load_mnist(test_batch_size):
    global train_loader
    global test_loader
    print('==> Preparing data..')

    transform_test = tv.transforms.Compose([
        tv.transforms.ToTensor(),
    ])

    testset = tv.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

    # loading the test data from testset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)


def load_fashion_mnist(test_batch_size):
    global train_loader
    global test_loader
    print('==> Preparing data.. Fashion-MNIST')
    testset = tv.datasets.FashionMNIST(root="./train_data", train=False, download=True, transform=tv.transforms.ToTensor())

    # loading the test data from testset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)


def load_cifar(test_batch_size):
    global train_loader
    global test_loader
    print('==> Preparing data..')
    data_dir = '/home/ahmsoy00/Projects/BNN/Scrub_with_FashionMNIST_CNN/data/cifar3'
    # data_dir ='E:\KIT\Project\CIFAR3.Scrub\data\cifar3' # local

    test_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(test_batch_size),
            tv.transforms.ToTensor(),
        ])

    print(os.listdir(data_dir))

    test_classes = os.listdir(data_dir + "/test")
    print('test Classes: ', test_classes)

    testset = ImageFolder(data_dir + '/test', transform=test_transform)
    print(testset.classes)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


def SAC(weight, d):
    with torch.no_grad():
        ones = weight.sign().add(1).sign().triu(d).sum()
        if not args.no_cuda:
            total = torch.ones(weight.shape).triu(d).sum().cuda()
        else:
            total = torch.ones(weight.shape).triu(d)
        percent = np.round((ones / total).mul(100).item(), 3)
    return percent


def SAC_(weight, d):
    with torch.no_grad():
        min_ones = weight.sign().sub(1).sign().tril(d - 1).sum().abs()
        if not args.no_cuda:
            total = torch.ones(weight.shape).tril(d - 1).sum().cuda()
        else:
            total = torch.ones(weight.shape).tril(d - 1).sum()
        percent = np.round((min_ones / total).mul(100).item(), 3)
    return percent


def test(model, criterion):
    model.eval()
    test_loss, correct, total, predicted, test_data = 0, 0, 0, 0, None
    global test_loader
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

            out = model(data)

            test_loss += criterion(out, target).item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)

            correct += (predicted == target).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100 * (correct / total)
    return accuracy, test_loss


def show_coverage_one_layer(model, scrubbed):
    if scrubbed:
        print('Scrubbed coverage:- ')
    print('layer ', args.layer, ' SAC: ', SAC(weight=model.classifier[args.layer].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[args.layer].weight, d=args.d))


def show_coverage_all_hidden(model, scrubbed):
    if scrubbed:
        print('Scrubbed coverage:- ')
    print('layer ', 1, '  SAC: ', SAC(weight=model.classifier[3].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[3].weight, d=args.d))
    print('layer ', 2, ' SAC: ', SAC(weight=model.classifier[6].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[6].weight, d=args.d))
    print('layer ', 3, ' SAC: ', SAC(weight=model.classifier[9].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[9].weight, d=args.d))
    print('layer ', 4, ' SAC: ', SAC(weight=model.classifier[12].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[12].weight, d=args.d))


def evaluate_ret_fault(model):
    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Best acc ', checkpoint['test_accuracy'])

    print('Initial Coverage:-')
    if args.layer != -1:
        """
        FI into only one layer
        """
        show_coverage_one_layer(model, False)
    else:
        show_coverage_all_hidden(model, True)

    fi = RetentionFault()

    print("Evaluating Retention Failures in Ref model:---------------------")
    results_acc = []
    results_acc_run = []
    life = [0]

    test_acc0, test_loss = test(model, criterion)
    results_acc_run.append(np.round(test_acc0, 2))
    test_acc0 = np.round(test_acc0, 2)
    print("Initial accuracy: {} test acc {:.3f}".format(1, test_acc0))

    for run in range(0, args.redundant_runs):

        print("Redundant runs: {} d1: {}".format(run, args.delta))
        life = [0]
        model.load_state_dict(checkpoint['model_state_dict'])  # reset weights
        results_acc_run = [test_acc0]

        for i in range(1, args.iterations + 1):

            if args.layer != -1:
                model = fi.inject_fault_with_delta_mask(model, args.t, args.layer, args.delta)

                test_acc, test_loss = test(model, criterion)
                results_acc_run.append(np.round(test_acc, 2))
                life.append(i * 10)
                print("Expected operational time: {} test acc {:.3f}".format(i * 10, test_acc))
                show_coverage_one_layer(model, False)

                """
                Scrub
                """
                print('Scrubbing---------')
                model = fi.scrub_fully_connected(model, args.layer, args.d)
                test_acc, test_loss = test(model, criterion)
                results_acc_run.append(np.round(test_acc, 2))
                life.append((i + 1) * 10)
                print("{:.3f} accuracy after scrub at expected operational time: {}".format(test_acc, i * 10))
                show_coverage_one_layer(model, True)
                print()
            else:
                model = fi.inject_fault_with_delta_mask_all_layer_mlp(model, args.t, args.delta)

                test_acc, test_loss = test(model, criterion)
                results_acc_run.append(np.round(test_acc, 2))
                life.append(i * 10)
                show_coverage_all_hidden(model, False)
                """
                Scrub
                """
                print('Scrubbing---------')
                model = fi.scrub_hidden_fully_connected(model, args.d)
                test_acc, test_loss = test(model, criterion)
                results_acc_run.append(np.round(test_acc, 2))
                life.append((i + 1) * 10)
                print("{:.3f} accuracy after scrub at expected operational time: {}".format(test_acc, i * 10))
                show_coverage_all_hidden(model, True)
                print()

        results_acc.append(results_acc_run)
        print('At run: ', run, ' accuracies_' + str(args.delta) + ' = ', results_acc_run)
        print('')

    mean_acc = torch.Tensor(results_acc).mean(0)
    std_acc = torch.Tensor(results_acc).std(0)
    print('accuracies_' + str(args.delta) + ' = torch.', mean_acc)
    print('accuracies_' + str(args.delta) + '_std = torch.', std_acc)
    # print('mean ', mean_acc, ' std ', std_acc)
    print('life = ', list(life))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    parser = argparse.ArgumentParser(description='NeuroScrub Simulation')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='N', help='input batch size for testing (default: 128)')
    parser.add_argument('--d', type=float, default=9, metavar='D', help='diagonal (default: 9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 100)')
    parser.add_argument('--NN_type', action='store', default='MLP', help='NN type (default:LeNet_5)')
    parser.add_argument('--dataset', action='store', default='fashionmnist', help='dataset (default:MNIST)')
    parser.add_argument('--pretrained_path', action='store_true', default='trained_weights/acc_mnist.pt', help='whether to run evaluation')
    parser.add_argument('--scrub_freq', type=float, default=1, help='scrub frequency (default:1y)')
    parser.add_argument('--delta', type=float, default=40, help='thermal stability factor(default:30)')
    parser.add_argument('--layer', type=float, default=3, help='Fault injection layer factor(default:H1). Choose -1 for all')
    parser.add_argument('--iterations', type=float, default=10, help='how many FI per year/month(default:10%)')
    parser.add_argument('--redundant_runs', type=float, default=1, help='how many fault run are done to find the average accuracy(default:1)')
    parser.add_argument('--t', type=float, default=365 * 24 * 60 * 60, help='the observed time interval(default:365 * 24 * 60 * 60 s)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train_loader = None
    test_loader = None
    """
    specify diagonal
    """
    diagonal = args.d

    print(args)

    """
    Define default input channel and number of classes
    """
    c_in, num_classes = 1, 10

    """
    select dataset
    """
    if str(args.dataset).casefold() == 'mnist':
        load_mnist(args.test_batch_size)
        c_in, num_classes = 1, 10
    elif str(args.dataset).casefold() == 'fashionmnist' or str(args.dataset).casefold() == 'fashion-mnist':
        load_fashion_mnist(args.test_batch_size)
        c_in, num_classes = 1, 10
    elif str(args.dataset).casefold() == 'cifar-3' or str(args.dataset).casefold() == 'cifar3':
        load_cifar(args.test_batch_size)
        c_in, num_classes = 3, 3

    """
    select model
    """
    if str(args.NN_type).casefold() == 'mlp':
        model = BinMLP(c_in, num_classes)
    elif str(args.NN_type) == 'cnn':
        model = BinLeNet(c_in, num_classes)

    """ select seed to reproduce results"""
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    evaluate_ret_fault_one_layer(model)
