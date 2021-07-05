"""
Author: Soyed Tuhin Ahmed
"""
import torch
import torchvision as tv
import numpy as np
import os
from bin_model import *
from float_model import *
# from retention_fault import *
import argparse
from torchvision.datasets import ImageFolder


def load_mnist(batch_size, test_batch_size):
    global train_loader
    global test_loader
    print('==> Preparing data..')
    transform_train = tv.transforms.Compose([
        tv.transforms.ToTensor(),
    ])
    transform_test = tv.transforms.Compose([
        tv.transforms.ToTensor(),
    ])

    trainset = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
    testset = tv.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # loading the test data from testset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)


def load_fashion_mnist(batch_size, test_batch_size):
    global train_loader
    global test_loader
    print('==> Preparing data.. Fashion-MNIST')
    trainset = tv.datasets.FashionMNIST(root="./train_data", train=True, download=True, transform=tv.transforms.ToTensor())
    testset = tv.datasets.FashionMNIST(root="./train_data", train=False, download=True, transform=tv.transforms.ToTensor())

    # loading the training data from trainset
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # loading the test data from testset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)


def load_cifar(batch_size, test_batch_size):
    global train_loader
    global test_loader
    print('==> Preparing data..')
    data_dir = '/home/ahmsoy00/Projects/BNN/Scrub_with_FashionMNIST_CNN/data/cifar3'
    # data_dir ='E:\KIT\Project\CIFAR3.Scrub\data\cifar3' # local
    train_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(batch_size),
            tv.transforms.ToTensor(),
        ])

    test_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(test_batch_size),
            tv.transforms.ToTensor(),
        ])

    print(os.listdir(data_dir))

    train_classes = os.listdir(data_dir + "/train")
    print('Train Classes: ', train_classes)

    test_classes = os.listdir(data_dir + "/test")
    print('test Classes: ', test_classes)

    trainset = ImageFolder(data_dir + '/train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = ImageFolder(data_dir + '/test', transform=test_transform)
    print(testset.classes)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


def train(model, optimizer, criterion, lambda_):  # 1
    model.train()
    correct, total = 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

        optimizer.zero_grad()

        out = model(data)

        loss = criterion(out, target)

        penalty = model.classifier[3].weight[[h1_non_scrub_row, h1_non_scrub_col]].relu().pow(2).sum().div(norm_1) + \
                  model.classifier[3].weight.mul(-1)[[h1_scrub_row, h1_scrub_col]].relu().pow(2).sum().div(norm_1) + \
                  model.classifier[6].weight[[h2_non_scrub_row, h2_non_scrub_col]].relu().pow(2).sum().div(norm_2) + \
                  model.classifier[6].weight.mul(-1)[[h2_scrub_row, h2_scrub_col]].relu().pow(2).sum().div(norm_2) + \
                  model.classifier[9].weight[[h3_non_scrub_row, h3_non_scrub_col]].relu().pow(2).sum().div(norm_3) + \
                  model.classifier[9].weight.mul(-1)[[h3_scrub_row, h3_scrub_col]].relu().pow(2).sum().div(norm_3) + \
                  model.classifier[12].weight[[h4_non_scrub_row, h4_non_scrub_col]].relu().pow(2).sum().div(norm_4) + \
                  model.classifier[12].weight.mul(-1)[[h4_scrub_row, h4_scrub_col]].relu().pow(2).sum().div(norm_4)

        loss = loss + penalty.mul(lambda_).div(4)

        _, predicted = torch.max(out.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        optimizer.zero_grad()
        loss.backward()

        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)

        optimizer.step()

        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-0.99, 0.99))

        optimizer.zero_grad()

    return 100 * (correct / total)  # return accuracy


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


def show_coverage(model):
    print('layer ', 1, '  SAC: ', SAC(weight=model.classifier[3].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[3].weight, d=args.d))
    print('layer ', 2, ' SAC: ', SAC(weight=model.classifier[6].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[6].weight, d=args.d))
    print('layer ', 3, ' SAC: ', SAC(weight=model.classifier[9].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[9].weight, d=args.d))
    print('layer ', 4, ' SAC: ', SAC(weight=model.classifier[12].weight, d=args.d),
          ' SAC_ : ', SAC_(weight=model.classifier[12].weight, d=args.d))


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


def train_model(model, epochs, optimizer, criterion, resume=False, path='trained_weights/acc_mnist'):
    trained_data_path = '../trained_weights/'
    train_data_name = 'mnist_f.pt'
    best_acc, train_acc = 0, 0
    if resume:
        print('Resuming from: ', str(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Best acc ', checkpoint['test_accuracy'])
        best_acc = checkpoint['test_accuracy']

    for epoch in range(1, epochs):

        train_acc = train(model=model, optimizer=optimizer, criterion=criterion, lambda_=args.penalty_rate)

        test_acc, test_loss = test(model=model, criterion=criterion)

        if best_acc < test_acc:
            print('Best accuracy updated: saving model--------------------')
            show_coverage(model)
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': best_acc
            }, trained_data_path + 'acc_' + train_data_name)
        elif epoch % args.log_interval == 0 and epoch > 1:
            print('Saving model at checkpoint ---------------------- ')
            show_coverage(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': best_acc
            }, trained_data_path + train_data_name)

        print("Epochs {} Train acc: {:.4f} Test accuracy: {:.4f} Best accuracy: {:.4f} -----------".format(epoch, train_acc, test_acc, best_acc))

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': best_acc
    }, trained_data_path + train_data_name)


if __name__ == '__main__':

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    parser = argparse.ArgumentParser(description='NeuroScrub Simulation')
    parser.add_argument('--batch_size', type=int, default=256,  help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=512,  help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,  help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.06,  help='learning rate (default: 0.006)')
    parser.add_argument('--d', type=float, default=0,  help='diagonal (default: 9)')
    parser.add_argument('--penalty_rate', '--pr', default=1e1, type=float, help='penalty rate (default: 1e1)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, help='random seed (default: 100)')
    parser.add_argument('--log_interval', type=int, default=5, help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', action='store', default='mnist', help='dataset (default:MNIST)')
    parser.add_argument('--pretrained', action='store', default=False, help='pretrained model')
    parser.add_argument('--pretrained_path', action='store_true', default='trained_weights/mnist_f.pt', help='whether to run evaluation')
    parser.add_argument('--bin', action='store_true', default=False, help='BNN training (default: True)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train_loader = None
    test_loader = None

    """
    specify diagonal
    """
    diagonal = args.d

    penalty_rate = args.penalty_rate

    print(args)


    """
    Define default input channel and number of classes
    """
    c_in, num_classes = 1, 10

    """
    select dataset
    """
    if str(args.dataset).casefold() == 'mnist':
        print('mnist')
        load_mnist(args.batch_size, args.test_batch_size)
    elif str(args.dataset).casefold() == 'fashionmnist' or str(args.dataset).casefold() == 'fashion-mnist':
        load_fashion_mnist(args.batch_size, args.test_batch_size)
    elif str(args.dataset).casefold() == 'cifar-3' or str(args.dataset).casefold() == 'cifar3':
        load_cifar(args.batch_size, args.test_batch_size)
        c_in, num_classes = 3, 3

    """
    select model
    """
    if args.bin:
        model = BinMLP(c_in, num_classes)
    else:
        model = FloatMLP(c_in, num_classes)

    """ select seed to reproduce results"""
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()

    """
    define loss function
    """
    criterion = nn.CrossEntropyLoss()

    """
    define optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs

    """
    define shape of hidden layers
    """

    h1 = model.classifier[3].weight.shape
    h2 = model.classifier[6].weight.shape
    h3 = model.classifier[9].weight.shape
    h4 = model.classifier[12].weight.shape

    """
    define normalization constants
    """
    norm_1 = h1[0] * h1[1]
    norm_2 = h2[0] * h2[1]
    norm_3 = h3[0] * h3[1]
    norm_4 = h4[0] * h4[1]

    """
    generate scrub and non-scrub index
    """
    h1_scrub_row, h1_scrub_col = torch.triu_indices(h1[0], h1[1], diagonal)
    h2_scrub_row, h2_scrub_col = torch.triu_indices(h2[0], h2[1], diagonal)
    h3_scrub_row, h3_scrub_col = torch.triu_indices(h3[0], h3[1], diagonal)
    h4_scrub_row, h4_scrub_col = torch.triu_indices(h4[0], h4[1], diagonal)

    h1_non_scrub_row, h1_non_scrub_col = torch.tril_indices(h1[0], h1[1], diagonal-1)
    h2_non_scrub_row, h2_non_scrub_col = torch.tril_indices(h2[0], h2[1], diagonal-1)
    h3_non_scrub_row, h3_non_scrub_col = torch.tril_indices(h3[0], h3[1], diagonal-1)
    h4_non_scrub_row, h4_non_scrub_col = torch.tril_indices(h4[0], h4[1], diagonal-1)

    train_model(model, epochs, optimizer, criterion, resume=args.pretrained, path=args.pretrained_path)
