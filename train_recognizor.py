import datetime

import numpy as np
import tensorflow as tf

from model.crnn_model import CRNN
from recognizor_utils import params, char_dict, decode_to_text, data_generator, sparse_tuple_from

# initialize
iter = 1
training = True

# True if user wants to continue training from previous checkpoint
continue_training = False
model = CRNN(num_classes=params['NUM_CLASSES'], training=True)
#previous checkpoint directory
_ = [model.load_weights('checkpoints/model_default') if continue_training else True]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=5)

loss_ = []
loss_train = []
loss_test = []
accuracy = []
curr_accuracy = 0

# training
# dataset: https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth
# please change path in data_generator in recognizor_utils.py for accessing the dataset
# the full training set should containt 7224612 images / 64 = 112884 batches

total_case = 0
total_case_train = 0

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

epoch = 0

for x_batch, y_batch in data_generator(batches=112884, batch_size=64, epochs=10):

    # training ops
    indices, values, dense_shape = sparse_tuple_from(y_batch)
    y_batch_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    with tf.GradientTape() as tape:
        logits, raw_pred, rnn_out = model(x_batch, training=training)

        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_batch_sparse,
                                             logits=rnn_out,
                                             label_length=[len(i) for i in y_batch],
                                             logit_length=[params['SEQ_LENGTH']] * len(y_batch),
                                             blank_index=62))
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, grads)]

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(iter, loss)
    decoded_, _ = tf.nn.ctc_greedy_decoder(rnn_out,  # logits.numpy().transpose((1, 0, 2)),
                                           sequence_length=[params['SEQ_LENGTH']] * len(y_batch),
                                           merge_repeated=True)
    decoded_ = tf.sparse.to_dense(decoded_[0]).numpy()

    print([decode_to_text(char_dict, [char for char in np.trim_zeros(np.array(word), 'b')]) for word in (y_batch)[:4]])
    print([decode_to_text(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded_[:4]])

    train_loss = loss.numpy().round(1)

    loss_.append(loss.numpy().round(1))
    loss_train.append(loss.numpy().round(1))

    gt_train = [decode_to_text(char_dict, [char for char in np.trim_zeros(np.array(word), 'b')]) for word in (y_batch)]
    pre_train = [decode_to_text(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded_]

    total_case_train += len(gt_train)
    tp_case_train = 0
    for i in range(len(pre_train)):
        if (pre_train[i].lower() == gt_train[i].lower()):
            tp_case_train += 1

    # every i iterations, do the following:
    # save weights of the model
    # print current model results
    # check test set and its loss
    if iter % 100 == 0:

        decoded, log_prob = tf.nn.ctc_greedy_decoder(rnn_out,  # logits.numpy().transpose((1, 0, 2)),
                                                     sequence_length=[params['SEQ_LENGTH']] * len(y_batch),
                                                     merge_repeated=True)
        decoded = tf.sparse.to_dense(decoded[0]).numpy()
        print(iter, loss.numpy().round(1),
              [decode_to_text(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded[:4]])

        # loss_train.append(loss.numpy().round(1))
        with open('loss_train.txt', 'w') as file:
            [file.write(str(s) + '\n') for s in loss_train]

        # test loss on one batch of data
        for x_test, y_test in data_generator(batches=1, batch_size=124, epochs=1, dataset='test'):
            indices, values, dense_shape = sparse_tuple_from(y_test)
            y_test_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

            logits, raw_pred, rnn_out = model(x_test)
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_test_sparse,
                                                 logits=rnn_out,
                                                 label_length=[len(i) for i in y_test],
                                                 logit_length=[params['SEQ_LENGTH']] * len(y_test),
                                                 blank_index=62))
            test_loss = loss.numpy().round(1)
            loss_test.append(loss.numpy().round(1))

            decoded_test, _ = tf.nn.ctc_greedy_decoder(rnn_out,  # logits.numpy().transpose((1, 0, 2)),
                                                       sequence_length=[params['SEQ_LENGTH']] * len(y_test),
                                                       merge_repeated=True)
            decoded_test = tf.sparse.to_dense(decoded_test[0]).numpy()

            gt_ = [decode_to_text(char_dict, [char for char in np.trim_zeros(np.array(word), 'b')]) for word in
                   (y_test)]
            pre_ = [decode_to_text(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded_test]

            total_case += len(gt_)
            tp_case = 0
            for i in range(len(pre_)):
                if (pre_[i].lower() == gt_[i].lower()):
                    tp_case += 1

            print('tp_case: {0}'.format(tp_case))
            print('accuracy: {0}'.format(tp_case / len(gt_)))
            accuracy.append(tp_case / len(gt_))

            epoch += 1

            # use tensorboard to plot graphs
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('accuracy', tp_case_train / len(gt_train), step=epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss_test', test_loss, step=epoch)
                tf.summary.scalar('accuracy', tp_case / len(gt_), step=epoch)

            with open('loss_test.txt', 'w') as file:
                [file.write(str(s) + '\n') for s in loss_test]

            with open('accuracy.txt', 'w') as file:
                [file.write(str(s) + '\n') for s in accuracy]

            # Save model when the model gets a higher accuracy
            if tp_case / len(gt_) > curr_accuracy:
                curr_accuracy = tp_case / len(gt_)
                print('Save model {}'.format(iter))
                model.save_weights('checkpoints/new_checkpoint/model_default')
                # plt.figure(2)
                # plt.plot(range(len(loss_train)), loss_train, 'r')
                # plt.savefig('./fig/reco_{}.png'.format(iter))

    iter += 1
