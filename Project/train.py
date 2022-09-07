from densenet_utils import densenet_arg_scope
from densenet import densenet_121
import sys
import os.path
import time
import tensorflow as tf
import util as tu
import numpy as np
import threading
import tensorflow as tf
slim = tf.contrib.slim

"""	Procedure to train the model on ImageNet training set

            imagenet_path:	path to  train images,test images
            print_training_accuracy_after_steps: number representing how often printing the training accuracy
            print_validation_accuracy_after_steps: number representing how often make a test and print the validation accuracy
            trained_model_path: path where to save model's tensorflow checkpoint
            summary_path: path where to save logs for TensorBoard

    """


def train():
    #Multitheading is used to load the images in batches.
    threads_numbers = 5
    # dropout probability of 0.5 used in fully connected layers
    dropout = 0.5
    learning_rate = 0.001
    # 90 epochs , training process will run 90 time on all batches of images
    epochs = 50
    # images in one batch
    batch_size = 64
    # trained model will be saved here
    trained_model_path = './trained_model-densenet'
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    #logs files will be saved here that will be later used by tensorboard to show graphs
    summary_path = './summary'
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    # we have selecte 10 classed of imagenet dataset for our project. images of these classes are stored here
    imagenet_path = './images/10Classes/'
    # we will print loss and accuracy after 10 batches has been processed
    print_training_accuracy_after_steps = 10
    # we will validate and print loss and accuracy after 50 batches has been processed
    print_validation_accuracy_after_steps = 50
    train_img_path = os.path.join(imagenet_path, 'training_data')
    # total number of images in all classes
    ts_size = tu.imagenet_size(train_img_path)
    #number of batches
    num_batches = int(float(ts_size) / batch_size)
    #selected classes from imagenet dataset
    wnid_labels30 = ["n01440764",
                     "n01443537",
                     "n01484850",
                     "n01491361",
                     "n01494475",
                     "n01496331",
                     "n01498041",
                     "n01514668",
                     "n01514859",
                     "n01518878",
                     "n01530575",
                     "n01531178",
                     "n01532829",
                     "n01534433",
                     "n01560419",
                     "n01537544",
                     "n01806567",
                     "n01580077",
                     "n01608432",
                     "n01592084",
                     "n01601694",
                     "n01582220",
                     "n01614925",
                     "n01630670",
                     "n01616318",
                     "n01622779",
                     "n01629819",
                     "n01632777",
                     "n01632458",
                     "n01631663"]

    wnid_labels10 = ["n01440764",
                    "n01443537",
                    "n01484850",
                    "n01491361",
                    "n01494475",
                    "n01496331",
                    "n01498041",
                    "n01514668",
                    "n01514859",
                    "n01518878"
                    ]

    #placeholders that will be used later for loading images on defining dropout rate
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 10])
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # load batches in queue to be processed. this is First in first out queue (FIFO)
    q = tf.FIFOQueue(
        batch_size * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [10]])
    #load in queue
    enqueue_op = q.enqueue_many([x, y])
    #read from queue
    x_b, y_b = q.dequeue_many(batch_size)
   #model classifer , see AlexNetModel class for more
    with slim.arg_scope(densenet_arg_scope()):
            with slim.arg_scope(densenet_arg_scope()):
                prediction, end_points = densenet_121(
                    x_b, 10, output_stride=8, is_training=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y_b, name='cross-entropy'))
    # Attach scalar function with loss function to view output in tensorboard
    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    epoch = tf.div(global_step, num_batches)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss, global_step=global_step)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_b, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Attach scalar function with accuracy function to view output in tensorboard
    tf.summary.scalar('accuracy', accuracy)

    # merge summaries to write them to file
    merged = tf.summary.merge_all()

    # checkpoint saver
    saver = tf.train.Saver()

    #coordinate the termination of a set of threads for image batch threads
    coord = tf.train.Coordinator()

    # init all global variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # enqueuing batches procedure
        def enqueue_batches():
            while not coord.should_stop():
                im, l = tu.read_batch(batch_size, train_img_path, wnid_labels10)
                sess.run(enqueue_op, feed_dict={x: im, y: l})

        # creating and starting parallel threads to fill the queue
        num_threads = threads_numbers
        for _ in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()

        # operation to write logs for tensorboard visualization
        train_writer = tf.summary.FileWriter(
            os.path.join(summary_path, 'train'), sess.graph)

        start_time = time.time()
        for e in range(sess.run(epoch), epochs):
            for _ in range(num_batches):
                summary, _, step = sess.run([merged, optimizer, global_step], feed_dict={
                                            lr: learning_rate, keep_prob: dropout})
                train_writer.add_summary(summary, step)

                # display current training informations
                if step % print_training_accuracy_after_steps == 0:
                    temp_time = time.time()
                    loss_value, training_accuracy_value = sess.run([loss, accuracy], feed_dict={
                        lr: learning_rate, keep_prob: 1.0})
                    print("time: ", temp_time-start_time,
                          'Epoch: {:03d} Step/Batch: {:09d} --- Loss: {:.7f} Training accuracy: {:.4f}'.format(e, step, loss_value, training_accuracy_value))

                # make test and evaluate validation accuracy
                if step % print_validation_accuracy_after_steps == 0:
                    #val_im, val_cls = tu.read_validation_batch(batch_size, os.path.join(imagenet_path, 'testing_data'), os.path.join(imagenet_path, 'data/validation_ground_truth.txt'))
                    val_im, val_cls = tu.read_batch(100, os.path.join(
                        imagenet_path, 'validation_data'), wnid_labels10)
                    v_a = sess.run(accuracy, feed_dict={
                                   x_b: val_im, y_b: val_cls, lr: learning_rate, keep_prob: 1.0})
                    # intermediate time
                    int_time = time.time()
                    print('Elapsed time: {}'.format(
                        tu.format_time(int_time - start_time)))
                    print('Validation accuracy: {:.04f}'.format(v_a))
                    # save weights to file
                    save_path = saver.save(sess, os.path.join(
                        trained_model_path, 'densenet-cnn.ckpt'))
                    print('Variables saved in file: %s' % save_path)
        end_time = time.time()
        print('Elapsed time: {}').format(tu.format_time(end_time - start_time))
        save_model_path = saver.save(sess, os.path.join(
            trained_model_path, 'densenet.trained_model'))
        print('Model saved in file: %s' % save_model_path)
        #close summary writer
        train_writer.close()
        #close thread coordinator
        coord.request_stop()

#initate training
train()
