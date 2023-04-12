import torch
from torch.utils import collect_env
import re
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

#print(collect_env.main())



def decrpyt_lines(lines: list[str], num_estimators: int, num_epoch: int):
    
    division_in_string = lambda x : int(x.split('/')[0]) / int(x.split('/')[1])
    get_numeric = lambda x: division_in_string(x) if "/" in x else float(x)
    
    est_dict = {}
    
    for est in range(num_estimators):
        est_dict[est] = {}
        for ep in range(num_epoch):
            est_dict[est][ep] = {'loss': [], 'acc': []}
    
    for line in lines:
        tmp = line.split(" | ")
        keyval = {}
        for elem in tmp:
            key, val = elem.split(": ")
            keyval |= {key: get_numeric(val)}        
        est_dict[keyval['Estimator']][keyval['Epoch']]['loss'].append(keyval['Loss'])
        est_dict[keyval['Estimator']][keyval['Epoch']]['acc'].append(keyval['Correct'])
        
        del keyval

    return est_dict

def get_means(estDict: dict):

    out = dict.fromkeys(estDict.keys(), {})

    for est, inner in estDict.items():
        out[est]['mean_loss'] = []
        out[est]['mean_acc'] = []
        for ep, tmp in inner.items():
            if len(tmp['loss']) > 0:
                out[est]['mean_loss'].append(np.array(estDict[est][ep]['loss'], dtype=float).mean())
                out[est]['mean_acc'].append(np.array(estDict[est][ep]['acc'], dtype=float).mean())
    
    estDict |= out


def plot_and_get_results(logfile: str):
    
    with open('./test.log', 'r') as f:
        lines = f.readlines()
        
    validation_lines = [line for line in lines if re.search(r"Validation Acc", line)]
    validation_acc = [float(re.search(r"Validation Acc: ([\d.]+)", line).group(1)) for line in validation_lines ]
    
    lines = [line for line in lines if re.match(r"^Estimator.*", line)]        
    info = decrpyt_lines(lines, 10, 50)
    get_means(info)
    
    
    num_estimators = len(info.keys())
    
    est_losses = []    
    est_acc = []
    
    # Plot and save
    plt.figure(figsize=(12, 8), num=1)
    plt.clf()
    
    for est, val in info.items():
        est_losses.append(val['mean_loss'][-1])
        plt.plot(range(1, len(val['mean_loss']) + 1), val['mean_loss'], label=f'Estimator-{str(est)} Loss')

    plt.legend()
    plt.grid()
    plt.title('Training Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('outputs/01-loss.pdf')

    plt.figure(figsize=(12, 8), num=2)
    plt.clf()
    plt.show()

    for est, val in info.items():
        est_acc.append(val['mean_acc'][-1])
        plt.plot(range(1, len(val['mean_acc']) + 1), np.array(val['mean_acc'], dtype=float) * 100, label=f'Estimator-{str(est)} Acc')

    plt.plot(range(2, len(validation_acc) + 2), validation_acc, label=f"Validation Acc")

    plt.legend()
    plt.grid()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('outputs/02-accuracy.pdf')
    plt.show()

    return sum(est_losses)/num_estimators, sum(est_acc)/num_estimators, validation_acc[-1]


if __name__ == "__main__":    
    #plot(logfile='./test.log')

    import glob
    
    name = glob.glob("./weights/*")[0]

    state = torch.load(name)

    print(state['model']['estimators_.0.convolutions.0.weight'].shape)