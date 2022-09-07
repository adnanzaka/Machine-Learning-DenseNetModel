import tensorflow as tf
import sys
import os.path
slim = tf.contrib.slim
from densenet import densenet_121
from densenet_utils import densenet_arg_scope
import util as tu
 # images in one batch
batch_size = 8
# trained model will be saved here
trained_model_path = './trained_model-Densenet'
if not os.path.exists(trained_model_path):
    os.makedirs(trained_model_path)
#logs files will be saved here that will be later used by tensorboard to show graphs
summary_path = './summary'
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
# we have selecte 10 classed of imagenet dataset for our project. images of these classes are stored here
imagenet_path = './images/'
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
wnid_labels = ["n01440764", "n01443537", "n01514668", "n01514859",
    "n01518878", "n01530575", "n01531178", "n01537544", "n01631663", "n01632458"]


inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
inputs_ = tf.reshape(inputs, [-1, 224, 224, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
with slim.arg_scope(densenet_arg_scope()):
    with slim.arg_scope(densenet_arg_scope()):
        logits, end_points = densenet_121(inputs_, 10, output_stride=8, is_training=True)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Attach scalar function with accuracy function to view output in tensorboard  
tf.summary.scalar('accuracy', accuracy)

# merge summaries to write them to file
merged = tf.summary.merge_all()

# checkpoint saver
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        # operation to write logs for tensorboard visualization
    train_writer = tf.summary.FileWriter(
        os.path.join(summary_path, 'train'), sess.graph)

    for i in range(20000):
        im, l = tu.read_batch(batch_size, train_img_path, wnid_labels)
        summary, _ = sess.run([merged, train_step], feed_dict={
                                            inputs: im, y_: l})
        train_writer.add_summary(summary, i)
        
        train_accuracy = accuracy.eval(feed_dict={
                inputs: im, y_: l})
        print('step %d, training accuracy %g' % (i, train_accuracy))

