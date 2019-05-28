import tensorflow as tf

if __name__ == '__main__':
    """1.placeholder 是 Tensorflow的占位符，暂时存放数据,
    如果需要从外部导入data，需要用到tf.placeholder()"""
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)

    """2. tf.multiply() ：两个矩阵中对应元素相乘
          tf.matmul() ：两个矩阵相乘"""
    result = tf.multiply(x1,x2)
    with tf.Session() as sess:
        print(sess.run(result,feed_dict={x1:[7.0],x2:[2.0]}))
