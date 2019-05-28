import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
import numpy as np

def add_layer(inputs,in_size,out_size,n_layer,activation_function = None,norm = None):
    layer_name = "layer_%s"% n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal(shape=[in_size,out_size]),name="W")
            tf.summary.histogram(layer_name+"/weights",weights)
        with tf.name_scope("bais"):
            bais = tf.Variable(tf.zeros([1,out_size])+0.1,name="b")
            tf.summary.histogram(layer_name + "/bais", bais)
        with tf.name_scope("result"):
            result = tf.add(tf.matmul(inputs,weights),bais)
        if norm:
            # tf.nn.moments 返回的mean是一阶矩（期望），variance是二阶中心矩（方差）
            fc_mean,fc_variance = tf.nn.moments(result,axes=[0])# axes = [0]表示batch维度。
            # 如果是图像数据，传入[0,1,2] ==> [batch,height,width]的均值/方差
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            norm_result = tf.nn.batch_normalization(result,fc_mean,fc_variance,shift,scale,epsilon)
            # 上面那一步, 在做如下事情:
            # Wx_plus_b = (result - fc_mean) / tf.sqrt(fc_variance + 0.001)
            # Wx_plus_b = result * scale + shift
            if activation_function is None:
                outputs = norm_result
            else:
                outputs = activation_function(norm_result)
            tf.summary.histogram(layer_name + "/outputs",outputs)
            return outputs
        else:
            if activation_function is None:
                outputs = result
            else:
                outputs = activation_function(result)
            tf.summary.histogram(layer_name + "/outputs",outputs)
            return outputs

def get_data():
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    return mnist

def compute_accuracy(test_xs,test_ys):
    global prediction
    y_prediction = sess.run(prediction,feed_dict={xs:test_xs})
    # tf.equal(A，B)返回的是和A一样长度的false 或者是True（0，1）
    # tf.arg_max 返回最大值的索引，1是行，0是列。返回每一列最大值的索引
    compile_y = tf.equal(tf.arg_max(y_prediction,1),tf.arg_max(test_ys,1))
    # tf.cast(a,dtype) 将a转化维dtype格式
    accuracy = tf.reduce_mean(tf.cast(compile_y,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:test_xs,ys:test_ys})
    return result


if __name__ == '__main__':
    """导入数据"""
    mnist = get_data()

    with tf.name_scope("input"):
        xs = tf.placeholder(tf.float32,[None,784],name="x_input")
        ys = tf.placeholder(tf.float32, [None, 10],name = "y_input")

    """搭建网络"""
    layer_1 = add_layer(xs,784,100,n_layer=1,activation_function=tf.nn.relu,norm=False)
    prediction = add_layer(layer_1,100,10,n_layer=2,activation_function=tf.nn.softmax,norm=False)
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
        tf.summary.scalar("cross_entropy",cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    """构建会话"""
    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession

    merged = tf.summary.merge_all()
    """ 将构建的模型用tensorboard看"""
    writer = tf.summary.FileWriter("logs_classify", sess.graph)
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(init)
    for step in range(10000):
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if step % 50 ==0:
            print("{} step :accuracy is {:.2%}".format(step,compute_accuracy(mnist.test.images,mnist.test.labels)))
            np_layer_1 = np.array(sess.run(layer_1,feed_dict={xs:batch_xs,ys:batch_ys})).reshape((1,-1))
            print(np_layer_1.shape)
            plt.ion()
            plt.plot(len(np_layer_1),np_layer_1)
            plt.show()
            # 这里如果第一层激活函数用relu，会再图中报错，因为再layer1和layer2中的权重全是0了。这里如果用relu，必须在激活函数前用batch nomalization
            # final_result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
            # writer.add_summary(final_result, step)
