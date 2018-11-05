import tensorflow as tf
import numpy as np
import os
import argparse
from sklearn.utils import shuffle
from sklearn import preprocessing

#set up argparse to take flags $train and $test for which model_selection
#specify folders of mgc files as argument
#train takes and stores a model from a folder
#test takes a folder of generated mgcs and a model location

def open_binary_file(self, file_name, dimensionality):
    mgcfile = open(file_name, 'rb')
    contents = np.fromfile(mgcfile, dtype=np.float32)
    mgfile.close()
    assert contents.size % float(dimensionality) == 0.0,'specified dimension %s not compatible with data'%(dimensionality)
    contents = contents[:(dimensionality * (contents.size / dimensionality))]
    contents = contents.reshape((-1, dimensionality))
    return contents

#DATA = import the data - large list of frames
def get_data(folder):
        all_frames=[]
        all_pm=[]
        for file in os.listdir(folder):
            contents = open_binary_file(file)
            for frame in contents:
                all_frames.append(frame[:-2])
                if args.train == True:
                    #Create 1-hot vector for class label. [1, 0] = PM. [0, 1] = not
                    if frame[-2] == 1:
                        all_pm.append([1, 0])
                    elif frame[-2] == 0:
                        all_pm.append([0, 1])
                    else:
                        raise ValueError, "This PM label should have value 1 or 0"
                    #all_pm.append(frame[-1])

        #NORMALISE DATA
        all_frames = preprocessing.scale(all_frames)
        #SHUFFLE
        all_frames, all_pm = shuffle(all_frames, all_pm)
        return all_frames, all_pm

#Training

def train_time(training_data):
    #Initialise architecture

    # Create the model
    x = tf.placeholder(tf.float32, [None, 248])
    W = tf.Variable(tf.zeros([248, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    frames, pms = get_data(training_data)

    # Train
    for _ in range(10):
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        #WRITE FUNCTION TO RETRIEVE BATCH. FOR NOW:
        batch_xs = frames[:1000]
        batch_ys = pms[:1000]
        test_x = frames[-300:]
        test_y = pms[-300:]

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        #Check acc after each epoch
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print "======================="
        print "Epoch " + str(_)
        print(sess.run(accuracy, feed_dict={x: test_x,
                                            y_: test_y}))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "======================================================="
    print "Final Accuracy"
    print(sess.run(accuracy, feed_dict={x: test_x,
                                        y_: test_y}))


#def test_time():
    #stuff
    #return


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/home/andrew/Documents/NoVoc_Djockovic/test/REF/',
                      help='Directory of mgc files')
  parser.add_argument('--train', '-t', action='store_true', default=True, help='Train from given input directory'
  ## I AM UP TO HERE
  args = parser.parse_args()
  if args.train == True:
    tf.app.run(main=train_time, argv=args.data_dir)
