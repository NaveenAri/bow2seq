import os
import random
from time import time, ctime
import re
import numpy as np
from pprint import pformat
import json
import joblib
from tqdm import tqdm
import tensorflow as tf
from model import Bow2Seq
from vocab import GloveVocab, SennaVocab, Vocab
from utils import Logger, AttrDict
#TODO word emebedding learning rate

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('bow_avg', False, 'average the words bow word embeddings to get encoding')
flags.DEFINE_integer('bow_layers', 2, 'number of feedforward layers to be applied to bow vector')
flags.DEFINE_integer('rnn_layers', 3, 'word embedding size')
flags.DEFINE_integer('embedding_size', 300, 'word embedding size')
flags.DEFINE_integer('hidden_size', 300, 'hidden size')
flags.DEFINE_boolean('share_embedding', False, 'reuse embedding from bow modue for decoder')

flags.DEFINE_float('l2', 0.0, 'l2 reg')
flags.DEFINE_float('dropout', 0.7, 'keep prob')
flags.DEFINE_float('embedding_dropout', 1.0, 'keep prob applied at bow and decoding')
flags.DEFINE_float('word_dropout', 1.0, 'fraction of whole words to drop out prior to computing vec representation of bow')
flags.DEFINE_integer('vocab_size', 10000, 'max vocab size')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_float('embedding_learning_rate', 0.0001, 'initial learning rate for word embeddings.')
flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'factor to multiply learning rate by to decay')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_string('embedding', 'random', 'choose from ["glove", "senna", "hybrid", "random"]')
flags.DEFINE_integer('max_sentence_len', 10, 'max length of sentence')
flags.DEFINE_string('dataset', 'ptb', 'choose from ["1bw", "ptb"]')

flags.DEFINE_string('early_stop', 'loss', 'early stoppping criterion ["accuracy","loss"]')
flags.DEFINE_integer('patience', 5, 'patience for early stopping')
flags.DEFINE_integer('decay_patience', 0, 'patience for decaying learning rate')
flags.DEFINE_integer('total_epochs', 32, 'max trainging epochs')
flags.DEFINE_boolean('test_only', False, 'only test already trained model')
flags.DEFINE_boolean('pre_trained', None, 'path to load pretrained model from')
flags.DEFINE_integer('train_size', 0, 'max items to use for trainig')
flags.DEFINE_integer('test_size', 0, 'max items to use for dev and test')
flags.DEFINE_integer('seed', 0, 'seed for rng')
flags.DEFINE_string('dest_path', 'temp_expts/%s'%re.sub(r'\s+|:', '_', ctime()).lower(), 'path to save model to')

GO ='_GO'
EOS = '_EOS'

def get_split(fname, max_sentence_len=0, max_items=0):
    lines = []
    with open(fname, 'r') as f:
        for line in f:
            #tokenize
            line = line.strip().split()
            #skip long sentence
            if max_sentence_len and len(line)>max_sentence_len:
                continue
            lines.append(line)
            if max_items and len(lines)>=max_items:
                break
    return lines

def padded(seqs, PAD_ID):
    #Pad sequences in the batch so they all have the same length as the max length sequence
    maxlen = max(map(lambda s: len(s), seqs))
    seqs = map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), seqs)
    return np.asarray(seqs)

def batches(seqs, batch_size, GO_ID, EOS_ID, PAD_ID, shuffle=False):
    if shuffle:
        random.shuffle(seqs)
    starts = range(0, len(seqs), batch_size)
    for start in tqdm(starts):
        end = start + batch_size
        seq_batch = seqs[start:end]
        #add go, eos, pad toks
        encoder_input = padded(seq_batch, PAD_ID)
        decoder_input = padded([[GO_ID]+seq for seq in seq_batch], PAD_ID)
        decoder_target = padded([seq+[EOS_ID] for seq in seq_batch], PAD_ID)
        #mask pad toks
        input_mask = (encoder_input != PAD_ID).astype(np.int32)
        output_mask = (decoder_target != PAD_ID).astype(np.int32)
        yield encoder_input, decoder_input, decoder_target, input_mask, output_mask

def calculate_metrics(session, seqs, model, config, GO_ID, EOS_ID, PAD_ID, train=False):
    losses = []
    correct = 0
    total = 0
    for batch in batches(seqs, config.batch_size, GO_ID, EOS_ID, PAD_ID, shuffle=train):
        encoder_input, decoder_input, decoder_target, input_mask, output_mask = batch
        loss_batch, correct_batch, total_batch = model.step(session, 
            encoder_input, decoder_input, decoder_target, input_mask, output_mask, train)
        losses.append(loss_batch)
        correct += correct_batch
        total += total_batch
    loss = float(sum(losses)) / len(losses)
    accuracy = float(correct) / total
    return {'loss': loss, 'acc': accuracy}

def run(_):
    FLAGS._parse_flags()
    config = AttrDict(FLAGS.__flags)
    np.random.seed(config.seed)
    tf.set_random_seed(config.seed)
    if not os.path.exists(config.dest_path):
        os.makedirs(config.dest_path)
    Logger.initialize(os.path.join(config.dest_path,'log.txt'))

    #load data
    train = get_split('data/%s/train.txt'%config.dataset, config.max_sentence_len, config.train_size)
    dev = get_split('data/%s/dev.txt'%config.dataset, config.max_sentence_len, config.test_size)
    test = get_split('data/%s/test.txt'%config.dataset, config.max_sentence_len, config.test_size)
    Logger.log('train(%d) dev(%d) test(%d)'%(len(train), len(dev), len(test)))

    #build vocab
    if config.embedding == 'glove':
        vocab = GloveVocab()
    elif config.embedding == 'senna':
        vocab = SennaVocab()
    else:
        vocab = Vocab()
    for sentence in train:
        vocab.add(sentence)
    if config.vocab_size:
        vocab = vocab.prune_rares_by_top_k(config.vocab_size-2)
    GO_ID, EOS_ID = vocab.add([GO, EOS])
    PAD_ID = vocab.PAD_INDEX
    Logger.log(str(vocab))

    config.vocab_size = len(vocab)

    Logger.log('numericalize...')
    train = [vocab.words2indices(sentence) for sentence in train]
    dev = [vocab.words2indices(sentence) for sentence in dev]
    test = [vocab.words2indices(sentence) for sentence in test]

    Logger.log('building model...')
    model = Bow2Seq(config)#, GO_ID, EOS_ID, PAD_ID)
    
    saver = tf.train.Saver()
    Logger.log('saving vocab and config')
    vocab.serialize(os.path.join(config.dest_path, 'vocab'))
    joblib.dump(config, os.path.join(config.dest_path, "config.pkl"))

    with open(os.path.join(config.dest_path, 'config.json'), 'wb') as f:
        json.dump(config, f, indent=2)
    Logger.log(pformat(config, indent=2))
    
    save_location = os.path.join(config.dest_path, "model.checkpoint")

    with tf.Session() as session:

        def save(scores):
            saver.save(session, save_location)
            with open(os.path.join(config.dest_path, 'best.json'), 'wb') as f:
                json.dump(scores, f, indent=2)
        # You can visualize the graph structure by running `tensorboard --logdir=/%(dest_path)/%(name)s/`
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        summary_op = tf.merge_all_summaries()
        split_summarywriter = {split: tf.train.SummaryWriter(os.path.join(config.dest_path, 'summary', split), flush_secs=5) for split in ['train', 'dev']}

        bestscores = {
            'dev': {
                'loss': float('inf'),
                'acc': 0,
            }
        }

        num_epochs_worse_than_best = 0

        if config.pre_trained:
            saver.restore(session, config.pre_trained)
        else:
            session.run(tf.initialize_all_variables())
            for var in tf.all_variables():
                if "Embedding" in var.name:
                    print var.name
            if config.embedding != 'random':
                Logger.log("Initialize embeddings")
                if config.embedding == 'glove':
                    word_embeddings = vocab.get_embeddings(corpus='wikipedia_gigaword', n_dim=config.embedding_size)
                elif config.embedding == 'senna':
                    assert config.embedding_size == 50
                    word_embeddings = vocab.get_embeddings()
                else:
                    raise NotImplementedError()
                for var in tf.all_variables():
                    if "Embedding" in var.name:
                        session.run(var.assign(word_embeddings))

        if not config.test_only:
            for epoch in xrange(config.total_epochs):
                train_start = time()
                bar = '*' * 10
                Logger.log('{} Epoch {} / {} {}'.format(bar, epoch, config.total_epochs, bar))

                train_results = calculate_metrics(session, train, model, config, GO_ID, EOS_ID, PAD_ID, train=True)
                dev_results = calculate_metrics(session, dev, model, config, GO_ID, EOS_ID, PAD_ID)
                scores = {
                    'epoch': epoch,
                    'train': train_results,
                    'dev': dev_results
                }
                
                #tf logging
                for split in ['train', 'dev']:
                    summary = tf.Summary()
                    if split == 'train':
                        summary.ParseFromString(session.run(summary_op))
                    summary.value.add(tag='loss', simple_value=scores[split]['loss'])
                    summary.value.add(tag='acc', simple_value=scores[split]['acc'])
                    split_summarywriter[split].add_summary(summary, epoch)

                #save best
                if config.early_stop == 'loss':
                    if scores['dev']['loss'] < bestscores['dev']['loss']:
                        bestscores = scores
                        save(scores)
                        num_epochs_worse_than_best = 0
                    else:
                        num_epochs_worse_than_best += 1
                elif config.early_stop == 'accuracy':
                    if scores['dev']['acc'] > bestscores['dev']['acc']:
                        bestscores = scores
                        save(scores)
                        num_epochs_worse_than_best = 0
                    else:
                        num_epochs_worse_than_best += 1
                else:
                    assert not config.patience, 'invalid early_stop option {}'.format(config.early_stop)

                if config.patience:
                    scores['early_stop'] = "{} out of {}".format(num_epochs_worse_than_best, config.patience)
                    if num_epochs_worse_than_best > int(config.patience):
                        Logger.log('early stopping on {} triggered'.format(config.early_stop))
                        break
                
                Logger.log(pformat(scores, indent=2))

                if config.decay_patience==0 or (num_epochs_worse_than_best>0 and num_epochs_worse_than_best%config.decay_patience==0):
                    Logger.log('learning_rate decayed to {}'.format(session.run([model.learning_rate_decay_op, model.embedding_learning_rate_decay_op])))

                Logger.log('finished in {}s'.format(time() - train_start))

        ###
        Logger.log('Testing...')
        saver.restore(session, save_location)
        test_results = calculate_metrics(session, test, model, config, GO_ID, EOS_ID, PAD_ID)
        bestscores['test'] = test_results
        Logger.log(pformat(config))
        Logger.log(pformat(bestscores))
        save(bestscores)


if __name__ == "__main__":
    main = run # For app run
    tf.app.run()