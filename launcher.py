"""Run multiple experiments in parallel with different parameters.

Usage:
    launcher.py
               [--expt_dir=<expt_dir>]
               [--gpus=<gpus>]
               [--autolaunch]
    launcher.py (-h | --help)
    launcher.py --version

Options:
    --expt_dir=<expt_dir>  [default: expts]
    --gpus=<gpus>
    --autolaunch
"""

import random
import subprocess
import time
import os
from docopt import docopt

values_ranges = {
    #model params
    "bow_avg": [False],#False works sightly better(evalued with 4 mil train after 10 epochs)
    "bow_layers": [3],
    "rnn_layers": [4],
    "embedding_size": [300],
    "hidden_size": [300],
    "share_embedding": [False],
    #training params
    "l2": [0.0],
    "dropout": [1.0],
    "embedding_dropout": [1.0],
    "word_dropout": [1.0],
    "vocab_size": [10000],
    "learning_rate": [0.001],
    "embedding_learning_rate": [0.0001],
    "learning_rate_decay_factor": [0.95],
    "batch_size": [512],
    "embedding": ['glove'],
    "max_sentence_len": [25],
    "dataset": ['1bw'],
    #expt params
    "early_stop": ['loss'],
    "patience": [5],
    "decay_patience": [0],
    "eval_granularity": [5],
    "total_epochs": [100],
    "test_only": [False],
    "train_size": [10000000],
    "test_size": [100000],
    "seed": [0],
}

def get_cmd(expt_dir):
    args = {}
    # randomly select values for experiment
    attempt=0
    while True:
        attempt+=1
        if attempt>10000:
            return
        args = {}
        for name, range_ in values_ranges.iteritems():
            args[name] = random.choice(range_)
        dest_path = '{}/data{}_{}_{},bow_avg{},bow_l{},rnn_l{},hid_s{},emb_s{},lr{},lr_dec{},lr_decpat{},drop{},emb{},share_emb{},maxlen{}'.format(expt_dir, args['dataset'], args['train_size'], args['test_size'], args['bow_avg'], args['bow_layers'], args['rnn_layers'], args['hidden_size'], args['embedding_size'], args['learning_rate'], args['learning_rate_decay_factor'], args['decay_patience'], args['dropout'], args['embedding'], args['share_embedding'], args['max_sentence_len'])
        
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            break
    args["dest_path"] = dest_path

    cmd = ["python", "run.py"]
    for name, value in args.iteritems():
        if type(value) != bool:
            cmd += ["--%s" % name, str(value)]
        elif value:
            cmd.append("--%s" % name)
        else:
            cmd.append("--no%s" % name)
    strcmd = ' '.join(cmd)
    return strcmd

if __name__ == '__main__':
    docopts = docopt(__doc__, version='Run Bow2Seq expts 0.1')
    gpus = docopts['--gpus'].split(',')
    if docopts['--autolaunch']:
        assert len(docopts['--gpus']) == 1, 'can only autolaunch with 1 gpus. Was given gpus=%s'%gpus
        while True:
            cmd = get_cmd(docopts['--expt_dir'])
            if cmd:
                print cmd
                cmd = 'CUDA_VISIBLE_DEVICES=%s %s'%(gpus[0], cmd)
                subprocess.call(cmd, shell=True)
            time.sleep(2)
    else:
        for gpu in gpus:
            cmd = get_cmd(docopts['--expt_dir'])
            assert cmd is not None
            print cmd
            cmd = 'CUDA_VISIBLE_DEVICES=%s %s'%(gpu, cmd)
            subprocess.Popen(cmd, shell=True)
