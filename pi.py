# coding: UTF-8

import tensorflow as tf

def PI():
    # BBP method
    def BBP(k_):
        result = tf.div((4./(8.*k_+1.)-1./(4.*k_+2.)-1./(8.*k_+5.)-1./(8.*k_+6.)), tf.pow(16., k_))
        return result

    # Variables
    sum_ = tf.Variable(0.)

    # Placeholders
    k_ = tf.placeholder("float32")
    output = tf.placeholder("float32")

    model = tf.initialize_all_variables()

    bbp_op = BBP(k_)
    sum_results = tf.assign(sum_, tf.add(sum_, output))

    with tf.Session() as sess:
        sess.run(model)

        for i in range(5):
            bbp_ = sess.run(bbp_op, feed_dict={k_: i})
            print("Accuracy: {0} BBPResult: {1}".format(bbp_, sess.run(sum_results, feed_dict={output: bbp_})))

if __name__ == '__main__':
    PI()

