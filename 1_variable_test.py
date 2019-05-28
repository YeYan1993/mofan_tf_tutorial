import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    """1.构建数据"""
    # 构建的数据是长度100的list，从（0，1】
    # 这里tf只支持float32类型的数据，因此需要转化
    input_data = np.random.rand(100).astype(np.float32)
    output_data = input_data * 0.1 + 0.3

    """2.构建模型"""
    # 构建变量，用的是tf.random_uniform(shape,min,max) 均匀分布【0，1）
    weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    bias = tf.Variable(tf.zeros([1]))
    y = weights * input_data + bias

    # 计算误差
    loss = tf.reduce_mean(tf.square(y - output_data))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    train = optimizer.minimize(loss)

    # 构建会话
    init = tf.global_variables_initializer() # 全局初始化
    sess = tf.Session()
    sess.run(init)

    # 训练20步打印权重和偏置
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print("Step {},weights is {} and bias is {}"
                  .format(step,sess.run(weights),sess.run(bias)))

