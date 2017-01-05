环境:

+ ubuntu16.04 
+ python==2.7.12
+ tensorflow==0.10.0rc0

> `tensorflow_py2` 是基于我的环境创建的VirtualEnv

---
## 基本使用
使用 TensorFlow, 你必须明白 TensorFlow:

+ 使用图 (graph) 来表示计算任务.
+ 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
+ 使用 tensor 表示数据.
+ 通过 变量 (Variable) 维护状态.
+ 使用 feed 和 fetch 为任意操作输入和输出数据.

## 综述
TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 `op `(operation 的缩写). 一个 op
获得 0 个或多个 `Tensor` , 执行计算, 产生 0 个或多个 Tensor .  
每个 `Tensor` 是一个类型化的多维数组. 例
如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 `[batch, height, width, channels]` .  
一个 TensorFlow 图描述了计算的过程. 为了进行计算, 图必须在 会话 里被启动. 会话 将图的 op 分发到
诸如 CPU 或 GPU 之类的 设备 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回. 在 P
ython 语言中, 返回的 tensor 是 numpy ndarray 对象; 在 C 和 C++ 语言中, 返回的 tensor 是 tensorflo
w::Tensor 实例.
## 计算图
TensorFlow 程序通常被组织成一个构建阶段, 和一个执行阶段. 在构建阶段, op 的执行步骤 被描述成一个图.  
在执行阶段, 使用会话执行执行图中的 op.  
例如, 通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op.  
TensorFlow 支持 C, C++, Python 编程语言. 目前, TensorFlow 的 Python 库更加易用, 它提供了大量的辅助  
函数来简化构建图的工作, 这些函数尚未被 C 和 C++ 库支持.
三种语言的会话库 (session libraries) 是一致的.
