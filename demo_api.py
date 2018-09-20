import tensorflow as tf
import numpy as np


# 演示API的类
class DemoTF:
    def __init__(self):
        '''''构造函数'''
        self.num1 = tf.Variable([1, 2, 0, 3], dtype=tf.int32)
        self.num2 = tf.Variable([1, 3, 0, 2], dtype=tf.int32)
        self.invert1 = tf.invert_permutation(self.num1)
        self.invert2 = tf.invert_permutation(self.num2)

    def run_graph(self):
        init = tf.global_variables_initializer()  # 初始化变量
        with tf.Session() as sess:
            sess.run(init)
            self.main(sess)  # 运行主函数

    def main(self, sess):
        '''''主函数'''
        print(sess.run(self.num1))
        print(sess.run(self.invert1))

        print('num2: \n', sess.run(self.num2))
        print(sess.run(self.invert2))

    # 主程序入口


if __name__ == "__main__":
    demo = DemoTF()
    demo.run_graph()