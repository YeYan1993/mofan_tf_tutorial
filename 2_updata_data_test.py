import tensorflow as tf

if __name__ == '__main__':
    state = tf.Variable(0,name="counter")

    # 定义常量
    one = tf.constant(1)

    # 定义加法步骤
    new_value = tf.add(state,one)

    # 将state更新（将new_value赋值更新给state）
    update = tf.assign(state,new_value)

    # 初始化所有变量（如果没有定义Variabel,就不需要了）
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))