import sys
import os
import joblib
import argparse
import logging
import numpy as np
import tensorflow as tf
from vocab import Vocab
from model import Bow2Seq

# def decode_greedy(model, sess, encoder_output):
#     decoder_state = None
#     decoder_input = np.array([data_utils.GO_ID, ], dtype=np.int32).reshape([1, 1])

#     attention = []
#     output_sent = []
#     while True:
#         decoder_output, attn_map, decoder_state = model.decode(sess, encoder_output, decoder_input, decoder_states=decoder_state)
#         attention.append(attn_map)
#         token_highest_prob = np.argmax(decoder_output.flatten())
#         if token_highest_prob == data_utils.EOS_ID or len(output_sent) > ARGS.max_seq_len:
#             break
#         output_sent += [token_highest_prob]
#         decoder_input = np.array([token_highest_prob, ], dtype=np.int32).reshape([1, 1])

#     return output_sent


def print_beam(beam, string='Beam'):
    print(string, len(beam))
    for (i, ray) in enumerate(beam):
        logging.debug(str((i, ray[0], ' '.join(ray[3]))))


def zip_input(beam):
    inp = np.array([ray[2][-1] for ray in beam], dtype=np.int32).reshape([-1, 1])
    return inp


def zip_state(beam):
    state = np.array([(ray[1]) for ray in beam])
    return state


def log_rebase(val):
    return np.log(10.0) * val


def beam_step(beam, candidates, decoder_output, zipped_state, vocab, max_beam_size, required_len=None):
    logprobs = (decoder_output).squeeze(axis=1) # squueze out time-axis to get [batch_size x vocab_size]
    newbeam = []

    for (b, ray) in enumerate(beam):
        prob, _, seq, low = ray
        for v in reversed(list(np.argsort(logprobs[b, :]))): # Try to look at high probabilities in each ray first

            newprob = prob + logprobs[b, v]

            newray = (newprob, zipped_state[b], seq + [v], low + [vocab.index2word(v)])
            # if v >= len(data_utils._START_VOCAB):
            #     newray = (newprob, zipped_state[b], seq + [v], low + [reverse_vocab[v]])
            # else:
            #     newray = (newprob, zipped_state[b], seq + [v], low)

            if len(newbeam) > max_beam_size and newprob < newbeam[0][0]:
                continue

            if (args.partial_sent or v == vocab.EOS_INDEX) and (not required_len or len(seq)==required_len):
                candidates += [newray]
                candidates.sort(key=lambda r: r[0])
                candidates = candidates[-max_beam_size:]
            if v != vocab.EOS_INDEX:
                newbeam += [newray]
                newbeam.sort(key=lambda r: r[0])
                newbeam = newbeam[-max_beam_size:]
    if len(candidates)>0:
        logging.debug('Candidates: %f - %f' % (candidates[0][0], candidates[-1][0]))
    else:
        logging.debug('Candidates: None')
    print_beam(newbeam)
    return newbeam, candidates


def beam_search(sess, model, encoding, vocab, max_beam_size, max_sent_len=50, required_len=None):
    state, output = None, None
    initial_state = np.tile(encoding, model.config.rnn_layers*2)
    beam = [(0.0, initial_state, [vocab.GO_INDEX], [''])] # (cumulative log prob, decoder state, [tokens seq], ['list', 'of', 'words'])

    candidates = []
    for i in xrange(max(max_sent_len, required_len)):#limit max decoder output length
        output, state = model.decode(sess, zip_input(beam), zip_state(beam))
        beam, candidates = beam_step(beam, candidates, output, state, vocab, max_beam_size, required_len)
        #TODO break after best ray is worse than best completed candidate?
        if len(candidates)>0 and beam[-1][0] < 1.5 * candidates[0][0]:
            logging.debug('Best ray is worse than worst completed candidate. candidates[] cannot change after this.')
            break

    print_beam(candidates, 'Final Candidates')
    finalray = candidates[-1]
    return finalray[3]
    #return finalray[2]


def translate_sent(sess, model, sentence, vocab):
    logging.debug('\n' + '*'*100 + '\n')
    sentence = sentence.lower().strip()
    logging.debug('input sent: %s'%sentence)
    #tokenize
    seq = vocab.words2indices(sentence.split())
    mask = np.ones_like(seq)
    logging.debug('seq %r'%seq)
    logging.debug('mask %r'%mask)
    # Encode
    feed = {
        model.encoder_input: [seq],
        model.input_mask: [mask],
        model.train: False
    }
    encoding = sess.run(model.rnn_initial_state, feed)
    encoding = encoding[0]#batch size = 1
    # Decode
    if args.matchlen:
        pred_words = beam_search(sess, model, encoding, vocab, max_beam_size=8, required_len=len(seq))
    else:
        pred_words = beam_search(sess, model, encoding, vocab, max_beam_size=8)
    pred_sentence = ' '.join(pred_words)
    logging.debug('predicted sent: %s'%pred_sentence)
    # Return
    return pred_sentence


def decode(model_dir, fname):
    vocab = Vocab.deserialize(os.path.join(model_dir, 'vocab'))
    config = joblib.load(os.path.join(model_dir, 'config.pkl'))
    with tf.Session() as session:
        model = Bow2Seq(config)
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(model_dir, 'model.checkpoint'))
        with open(fname, 'r') as fin:
            with open('%s.out'%fname, 'w', 0) as fout:
                for line in fin:
                    fout.write(translate_sent(session, model, line, vocab) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run beam search decoder given model checkpoint and input file'
    )
    parser.add_argument("model_dir")
    parser.add_argument("input")
    parser.add_argument("--matchlen", help="force decode sentence of same length as input",
                        action="store_true")
    parser.add_argument("--partial_sent", help="only consider complete sentences",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")


    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    decode(args.model_dir, args.input)
