import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

class Bow2Seq(object):
    def __init__(self, config):#, GO_ID, EOS_ID, PAD_ID):
        self.config = config
        # self.GO_ID = GO_ID
        # self.EOS_ID = EOS_ID
        # self.PAD_ID = PAD_ID
        #declare variables and placeholder
        encoder_input = tf.placeholder(tf.int32, shape=[None, None], name='encoder_input')
        decoder_input = tf.placeholder(tf.int32, shape=[None, None], name='decoder_input')
        decoder_target = tf.placeholder(tf.int32, shape=[None, None], name='decoder_target')
        input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_mask')
        output_mask = tf.placeholder(tf.int32, shape=[None, None], name='output_mask')
        train = tf.placeholder(tf.bool, shape=[], name='train')
        global_step = global_step = tf.Variable(0, trainable=False, name='global_step')
        dropout = tf.select(train, config.dropout, 1.0, name="dropout")
        embedding_dropout = tf.select(train, config.embedding_dropout, 1.0, name="dropout_word_embeddings")
        word_dropout = tf.select(train, config.word_dropout, 1.0, name="dropout_words")

        #build model
        encoder_input_embeddings = embedding_module(encoder_input, config.vocab_size, config.embedding_size, embedding_dropout)
        decoder_input_embeddings = embedding_module(decoder_input, config.vocab_size, config.embedding_size, embedding_dropout, reuse=True)
        encoding = bow_module(encoder_input_embeddings, input_mask, config.embedding_size, dropout, word_dropout, config.bow_layers, avg=config.bow_avg)
        decoder_initial_state = tf.tile(encoding, (1, 2*config.rnn_layers), name='decoder_initial_state')#using MultiLayer LSTM
        logits, decoder_state = decoder_module(decoder_input_embeddings, decoder_initial_state, output_mask,
                                config.hidden_size, config.vocab_size, config.rnn_layers,
                                dropout)
        pred = tf.argmax(logits, 2)
        logprobs, loss, correct, total = sequence_logprobs_loss_correct_total(logits, decoder_target, output_mask)

        #optimizer
        with tf.variable_scope('Optimizer'):
            l2_loss_elems = tf.get_collection('l2_loss')
            l2_loss = tf.add_n(l2_loss_elems) if len(l2_loss_elems)>0 else 0.0
            total_loss = loss + config.l2*l2_loss
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            we_optimizer = tf.train.AdamOptimizer(config.embedding_learning_rate)
            
            train_op = optimizer.minimize(total_loss, global_step=global_step, var_list=[var for var in tf.trainable_variables() if "Embedding" not in var.name])
            we_train_op = we_optimizer.minimize(total_loss, var_list=[var for var in tf.trainable_variables() if "Embedding" in var.name])
            train_op = tf.group(train_op, we_train_op)
            
        #placeholders
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.decoder_target = decoder_target
        self.input_mask = input_mask
        self.output_mask = output_mask
        self.train = train
        self.decoder_initial_state = decoder_initial_state

        #ops
        self.train_op = train_op
        self.encoding = encoding
        self.logprobs = logprobs
        self.pred = pred
        self.loss = loss
        self.correct = correct
        self.total = total
        self.global_step = global_step
        self.decoder_state = decoder_state

        #beam-search
        # beam_size = tf.placeholder(tf.int32)
        # beam_output, beam_scores = setup_beam(beam_size, L_dec, config.hidden_size, config.vocab_size, config.rnn_layers, GO_ID, EOS_ID)
        # self.beam_size = beam_size
        # self.beam_output = beam_output
        # self.beam_scores = beam_scores

    def step(self, session, encoder_input, decoder_input, decoder_target, input_mask, output_mask, train):
        feed = {
            self.encoder_input: encoder_input,
            self.decoder_input: decoder_input,
            self.decoder_target: decoder_target,
            self.input_mask: input_mask,
            self.output_mask: output_mask,
            self.train: train
        }
        if train:
            loss_batch, correct_batch, total_batch, _ = session.run([self.loss, self.correct, self.total, self.train_op], feed)
        else:
            loss_batch, correct_batch, total_batch = session.run([self.loss, self.correct, self.total], feed)
        return loss_batch, correct_batch, total_batch

    def decode(self, session, decoder_input, decoder_state):
        feed = {
            self.decoder_input: decoder_input,
            self.decoder_initial_state: decoder_state,
            self.output_mask: np.ones_like(decoder_input),
            self.train: False
        }
        seq_logprobs, decoder_state = session.run([self.logprobs, self.decoder_state], feed)
        return seq_logprobs, decoder_state

    def decode_beam(self, session, encoding, beam_size=8):
        feed = {
            self.encoding: encoding,
            self.train: False,
            self.beam_size: beam_size
        }
        seqs, scores = session.run([self.beam_output, self.beam_scores], feed)
        return seqs, scores

def embedding_module(token_seq, vocab_size, emb_size, dropout, scope="Embedding", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        L = tf.get_variable('Embedding', [vocab_size, emb_size])
        word_embeddings = tf.nn.embedding_lookup(L, token_seq)
        dropped_embeddings = tf.nn.dropout(word_embeddings, dropout)
    return dropped_embeddings

def bow_module(word_embeddings, token_mask, emb_size, dropout, word_dropout, layers=0, avg=False, scope="Encoder"):
    #emb_size = tf.shape(word_embeddings)[2]
    with tf.variable_scope(scope):
        token_mask = tf.nn.dropout(tf.to_float(token_mask), keep_prob=word_dropout)
        seq_len = tf.reduce_sum(token_mask, reduction_indices=1)
        word_embeddings_mask = tf.tile(tf.expand_dims(token_mask,2), (1,1,emb_size))
        word_embeddings = tf.select(tf.cast(word_embeddings_mask, tf.bool), word_embeddings, tf.zeros_like(word_embeddings))
        bow_vec = tf.reduce_sum(word_embeddings, reduction_indices=1, name='bow_sum')
        if avg:
            bow_vec = tf.truediv(bow_vec, tf.expand_dims(seq_len, 1), name='bow_avg')
        for layer_num in xrange(layers):
            with tf.variable_scope("layer_%d"%layer_num):
                W = tf.get_variable("W", [emb_size, emb_size])
                b = tf.get_variable("b", [emb_size])
                bow_vec = tf.nn.relu(tf.matmul(bow_vec, W) + b)
                bow_vec = tf.nn.dropout(bow_vec, keep_prob=dropout)
                tf.add_to_collection('l2_loss', tf.nn.l2_loss(W))
    return bow_vec

def decoder_module(input_embeddings, initial_state, mask, 
                   hidden_size, output_size, layers,
                   dropout, scope="Decoder"):
    with tf.variable_scope(scope) as scope:
        cell = rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=False)
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
        if layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers, state_is_tuple=False)
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, output_size)
        seq_len = tf.reduce_sum(mask, reduction_indices=1)
        logits, state = rnn.dynamic_rnn(cell, input_embeddings,
                                    sequence_length=seq_len, initial_state=initial_state,
                                    dtype=tf.float32, scope=scope)
    return logits, state

def sequence_logprobs_loss_correct_total(logits, labels, weights, scope="LossAndOutputProbs"):
    # logits(batch, time, num_classes)
    # labels(batch, time)
    # weights(batch, time)
    batch_size = tf.shape(logits)[0]
    max_time = tf.shape(logits)[1]
    num_classes = tf.shape(logits)[2]

    #rehaping since tensorflow doesn't support tensor ops
    logits_2d = tf.reshape(logits, [-1, num_classes])
    labels_1d = tf.reshape(labels, [-1])
    weights_1d = tf.reshape(weights, [-1])
    seq_len = tf.reduce_sum(weights, 1)
    #loss
    losses_1d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_2d, labels_1d) * tf.to_float(weights_1d)
    losses_2d = tf.reshape(losses_1d, tf.pack([batch_size, max_time]))
    mean_loss_per_seq = tf.reduce_sum(losses_2d, 1)/tf.to_float(seq_len)
    loss = tf.reduce_mean(mean_loss_per_seq)
    #logprobs
    logprobs_2d = tf.nn.log_softmax(logits_2d)
    logprobs = tf.reshape(logprobs_2d, [batch_size, max_time, num_classes])
    #correct
    correct_matrix = tf.to_int32(tf.equal(tf.argmax(logits, 2), tf.to_int64(labels))) * weights
    correct = tf.reduce_sum(correct_matrix)
    #total
    total = tf.reduce_sum(seq_len)
    return logprobs, loss, correct, total


# ###BEAM SEARCH###
# def decoder_graph(decoder_input_embedding, decoder_state, 
#                    hidden_size, output_size, layers, scope="Decoder"):
#     with tf.variable_scope(scope, reuse=True) as scope:
#         cell = rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
#         cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=1)
#         if layers > 1:
#             cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers, state_is_tuple=True)
#         cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, output_size)
#         decoder_output, decoder_state = cell(decoder_input_embedding, decoder_state)
#     return decoder_output, decoder_state

# def setup_beam(beam_size, L_dec, hidden_size, vocab_size, layers, GO_ID, EOS_ID):
#     time_0 = tf.constant(0)
#     beam_seqs_0 = tf.constant([[GO_ID]])
#     beam_probs_0 = tf.constant([0.])

#     cand_seqs_0 = tf.constant([[EOS_ID]])
#     cand_probs_0 = tf.constant([-3e38])

#     state_0 = tf.zeros([1, hidden_size])
#     states_0 = [(state_0, state_0)] * layers

#     def beam_cond(time, beam_probs, beam_seqs, cand_probs, cand_seqs, states):
#         return tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs)

#     def beam_step(time, beam_probs, beam_seqs, cand_probs, cand_seqs, states):
#         batch_size = tf.shape(beam_probs)[0]
#         inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [batch_size, 1]), [batch_size])
#         decoder_input = tf.nn.embedding_lookup(L_dec, inputs)
#         decoder_output, state_output = decoder_graph(decoder_input, states, hidden_size, vocab_size, layers)

#         with tf.variable_scope("Logistic", reuse=True):
#             do2d = tf.reshape(decoder_output, [-1, hidden_size])
#             logits2d = rnn_cell._linear(do2d, vocab_size, True, 1.0)
#             logprobs2d = tf.nn.log_softmax(logits2d)

#             total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
#             total_probs_noEOS = tf.concat(1, [tf.slice(total_probs, [0, 0], [batch_size, EOS_ID]),
#                                             tf.tile([[-3e38]], [batch_size, 1]),
#                                             tf.slice(total_probs, [0, EOS_ID + 1],
#                                                      [batch_size, vocab_size - EOS_ID - 1])])

#         flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
#         beam_k = tf.minimum(tf.size(flat_total_probs), beam_size)
#         next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

#         next_bases = tf.floordiv(top_indices, vocab_size)
#         next_mods = tf.mod(top_indices, vocab_size)

#         next_states = [tf.gather(state, next_bases) for state in state_output]
#         next_beam_seqs = tf.concat(1, [tf.gather(beam_seqs, next_bases),
#                                      tf.reshape(next_mods, [-1, 1])])

#         cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
#         beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
#         new_cand_seqs = tf.concat(0, [cand_seqs_pad, beam_seqs_EOS])
#         EOS_probs = tf.slice(total_probs, [0, EOS_ID], [batch_size, 1])
#         new_cand_probs = tf.concat(0, [cand_probs, tf.reshape(EOS_probs, [-1])])

#         cand_k = tf.minimum(tf.size(new_cand_probs), beam_size)
#         next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
#         next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

#         return [time + 1, next_beam_probs, next_beam_seqs, next_cand_probs, next_cand_seqs, next_states]

#     loop_vars = [time_0, beam_probs_0, beam_seqs_0, cand_probs_0, cand_seqs_0, states_0]
#     for var in loop_vars:
#         print var
#     ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, back_prop=False)
# #    time, beam_probs, beam_seqs, cand_probs, cand_seqs, _ = ret_vars
#     cand_seqs = ret_vars[4]
#     cand_probs = ret_vars[3]
#     return cand_seqs, cand_probs
