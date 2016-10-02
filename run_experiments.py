"""Run multiple experiments in parallel with different parameters.

Usage:
    run_experiments.py
               [--expt_dir=<expt_dir>]
               [--gpus=<gpus>]
               [--autolaunch]
    run_experiments.py (-h | --help)
    run_experiments.py --version

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
    "bow_avg": [False],
    "bow_layers": [0],
    "rnn_layers": [3],
    "embedding_size": [100],
    "hidden_size": [100],
    #training params
    "l2": [0.0],
    "dropout": [0.5],
    "embedding_dropout": [1.0],
    "word_dropout": [1.0],
    "vocab_size": [0],
    "learning_rate": [0.001],
    "embedding_learning_rate": [0.0001],
    "batch_size": [124],
    "embedding": ['glove'],
    "max_sentence_len": [15],
    #expt params
    "early_stop": ['loss'],
    "patience": [5],
    "total_epochs": [100],
    "test_only": [False],
    "pre_trained": [None],
    "max_items": [0],
    "seed": [0],
}

sofar = set()
def launch(expt_dir, gup_id, join=False):
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
        if str(args) not in sofar:
            dest_path = '{}/bow_avg{},bow_l{},rnn_l{},hid_s{},emb_s{},lr{},emb{},maxlen{}'.format(expt_dir, args['bow_avg'], args['bow_layers'], args['rnn_layers'], args['hidden_size'], args['embedding_size'], args['learning_rate'], args['embedding'], args['max_sentence_len'])
            if not os.path.exists(dest_path):
                sofar.add(str(args))
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

    strcmd = 'CUDA_VISIBLE_DEVICES=%s '%gup_id + ' '.join(cmd)
    print strcmd 
    if join:
        subprocess.call(strcmd, shell=True)
    else:
        subprocess.Popen(strcmd, shell=True)

if __name__ == '__main__':
    docopts = docopt(__doc__, version='Run MT_DMN expts 0.1')
    gpus = docopts['--gpus'].split(',')
    if docopts['--autolaunch']:
        assert len(docopts['--gpus']) == 1, 'can only autolaunch with 1 gpus. Was given gpus=%s'%gpus
        while True:
            time.sleep(2)
            launch(docopts['--expt_dir'], gpus[0], join=True)
    else:
        for gpu in gpus:
            launch(docopts['--expt_dir'], gpu)
