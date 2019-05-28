import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,n_layer,activation_function = None):
    layer_name = "layer%s"% n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal(shape=[in_size,out_size]),name="W")
            tf.summary.histogram(layer_name+"/weights",weights)
        with tf.name_scope("bais"):
            bais = tf.Variable(tf.zeros([1,out_size])+0.1,name="b")
            tf.summary.histogram(layer_name + "/bais", bais)
        with tf.name_scope("result"):
            result = tf.add(tf.matmul(inputs,weights),bais)
        if activation_function is None:
            outputs = result
        else:
            outputs = activation_function(result)
        tf.summary.histogram(layer_name + "/outputs",outputs)
        return outputs

if __name__ == '__main__':
    """导入数据"""
    x = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
    # 添加一个noise，让y更加靠近真实
    noise = np.random.normal(0,0.05,x.shape).astype(np.float32)
    y = np.square(x)-0.5 + noise  # np.square对array的每个元素求平方

    with tf.name_scope("input"):
        xs = tf.placeholder(tf.float32,[None,1],name="x_input")
        ys = tf.placeholder(tf.float32, [None, 1],name = "y_input")

    """搭建网络"""
    layer_1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.tanh)
    prediction = add_layer(layer_1,10,1,n_layer=2,activation_function=None)
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
        tf.summary.scalar("loss",loss)
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    init = tf.global_variables_initializer()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,y)
    plt.ion() # plt.show() 不暂停整个程序
    plt.show()

    """构建会话"""
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        """ 将构建的模型用tensorboard看"""
        writer = tf.summary.FileWriter("logs", sess.graph)
        sess.run(init)
        for step in range(1000):
            sess.run(train_step,feed_dict={xs:x,ys:y})
            if step % 50 ==0:
                print("{} step :loss is {}".format(step,sess.run(loss,feed_dict={xs:x,ys:y})))
                final_result = sess.run(merged,feed_dict={xs:x,ys:y})
                writer.add_summary(final_result,step)

                try:
                    # 首先需要移除生成的线，不然看不清
                    ax.lines.remove(lines[0])
                except Exception:
                    # 如果没有线，就跳过
                    pass
                predi_value = sess.run(prediction,feed_dict={xs:x,ys:y})
                lines = ax.plot(x,predi_value,'r-',lw = 5)
                plt.pause(0.1)


