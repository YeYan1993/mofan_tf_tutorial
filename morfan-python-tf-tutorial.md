# morfan-python-tf-tutorial
##  5_classify_test.py
#### 用mnist数据集实验：
* 1.三层网络：input ==> layer1(relu) ==> ouput(softmax) 结果：epoch10000，loss不会降低，acc = 9.8%，不会提升。
* 2.三层网络：input ==> layer1(None_activation) ==> ouput(softmax) 结果：epoch10000，loss不会降低，acc = 9.8%，不会提升。
* 3.三层网络：input ==> layer1(tanh) ==> ouput(softmax) 结果：epoch10000，acc 从10.66%到35%
* 4.三层网络：input ==> layer1(sigmoid) ==> ouput(softmax) 结果：epoch10000，acc从8.6%到43.4%
* 5.两层网络：input ==> ouput(softmax) 结果：epoch10000，acc从12%到53%。
* 6.三层网络：input ==> layer1 ==> Batch Nomalization ==> relu ==> output(softmax),loss降低，acc从13.5% 到 55%
#### 总结：
* 1.对于三层网络而言，就relu和None_activation效果都是一样，且loss得不到下降，理由是：relu可以看作为是对不加激活函数的一种dropout。
* 2.加了tanh和sigmoid激活函数之后，loss能降低，模型能学习，那么该分类更适用于非线性。
* 3.直接2层网络，模型能够更快的收敛，可能是数据简单了，2层网络就够用了，收敛速度肯定比三层更快。
## 11_BN_original_test.py
#### 实验过程
* 没有用 BN 的时候, 每层的值迅速全部都变为 0, 也可以说, 所有的神经元都已经死了. 而有 BN, relu 过后, 每层的值都能有一个比较好的分布效果, 大部分神经元都还活着. 因为没有使用 NB 的网络, 大部分神经元都死了, 所以连误差曲线都没了
* 把 relu 换成 tanh:可以看出, 没有 NB, 每层的值迅速全部都饱和, 都跑去了 -1/1 这个饱和区间, 有 NB, 即使前一层因变得相对饱和, 但是后面几层的值都被 normalize 到有效的不饱和区间内计算. 确保了一个活的神经网络.