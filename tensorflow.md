# 介绍
## 总览图片
<iframe src="http://www.tensorfly.cn/images/tensors_flowing.gif"></iframe>
## tensorflow运作方式入门(以MNIST例子)
### 准备数据
    数据集(MNIST)
    data_sets.train      55000个图像和标签(label),作为主要训练集
    data_sets.validation 5000个图像标签,用于迭代验证训练准确度
    data_sets.test       1000个图像和标签，用于最终验证训练准确度(train accuracy)

### 输入与占位符(input and Placeholders)
    placeholde_inputs()函数生成tf.placeholder操作，定义传入图表中的shape参数，shape参数包括batch_size
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
    IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
** 在训练循环（training loop）的后续步骤中，传入的整个图像和标签数据集会被切片，以符合每一个操作所设置的batch_size值，占位符操作将会填补以符合这个batch_size值。然后使用feed_dict参数，将数据传入sess.run()函数。**

## 构建图表
**在为数据创建占位符之后，就可以运行mnist.py文件，经过三阶段的模式函数操作：inference()， loss()，和training()。图表就构建完成了。**

### inference() (推理):函数会尽可能地构建图表，做到返回包含了预测结果（output prediction）的Tensor。它接受图像占位符为输入，在此基础上借助ReLu(Rectified Linear Units)激活函数，构建一对完全连接层（layers），以及一个有十个节点（node）、指明了输出logtis模型的线性层。每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀

tf.Variable实例:生成每一层所使用的权重和偏差，并且包含了各自期望的shape。
例如，当这些层是在hidden1作用域下生成时，赋予权重变量的独特名称将会是"hidden1/weights
每个变量在构建时，都会获得初始化操作（initializer ops）。
tf.truncated_normal函数初始化权重变量，给赋予的shape则是一个二维tensor，其中第一个维度代表该层中权重变量所连接（connectfrom）的单元数量，第二个维度代表该层中权重变量所连接到的（connect to）单元数量。对于名叫hidden1的第一层，相应的维度则是[IMAGE_PIXELS, hidden1_units]，因为权重变量将图像输入连接到了hidden1层。
tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布。
tf.zeros函数初始化偏差变量（biases),确保所有偏差的起始值都是0，而它们的shape则是其在该层中所接到的（connect to）单元数量。

```
with tf.name_scope('hidden1') as scope:
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS,hidden1_units],
                            stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
                            name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                        name='biases')
```

图表的三个主要操作，分别是两个tf.nn.relu操作，它们中嵌入了隐藏层所需的tf.matmul；以及logits模型所需的另外一个tf.matmul。三者依次生成，各自的tf.Variable实例则与输入占位符或下一层的输出tensor所连接

```
hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
logits = tf.matmul(hidden2, weights) + biases
```

### loss() 往inference图表中添加生成损失（loss）所需要的操作
*首先，labels_placeholer中的值，将被编码为一个含有1-hot values的Tensor。例如，如果类标识符为“3”，那么该值就会被转换为： 
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]*
```
batch_size = tf.size(labels)
labels = tf.expand_dims(labels, 1)
indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
concated = tf.concat(1, [indices, labels])
onehot_labels = tf.sparse_to_dense(
concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
```
之后，又添加一个tf.nn.softmax_cross_entropy_with_logits操作，用来比较inference()函数与1-hot标签所输出的logits Tensor。
```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                        onehot_labels,
                                                        name='xentropy')
```
然后，使用tf.reduce_mean函数，计算batch维度（第一维度）下交叉熵（cross entropy）的平均值，将将该值作为总损失。
```
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
#返回损失值的Tensor
```
## train() 往损失图表中添加计算并应用梯度（gradients）所需的操作
training()函数添加了通过梯度下降（gradient descent）将损失最小化所需的操作。
首先，该函数从loss()函数中获取损失Tensor，将其交给tf.scalar_summary，后者在与SummaryWriter（见下文）配合使用时，可以向事件文件（events file）中生成汇总值（summary values）。每次写入汇总值时，它都会释放损失Tensor的当前值（snapshot value）。
```
tf.scalar_summary(loss.op.name, loss)
```
接下来，我们实例化一个tf.train.GradientDescentOptimizer，负责按照所要求的学习效率（learning rate）应用梯度下降法（gradients）。
```
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
```
之后，我们生成一个变量用于保存全局训练步骤（global training step）的数值，并使用minimize()函数更新系统中的三角权重（triangle weights）、增加全局步骤的操作。根据惯例，这个操作被称为 train_op，是TensorFlow会话（session）诱发一个完整训练步骤所必须运行的操作（见下文）。
```
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
```
最后，程序返回包含了训练操作（training op）输出结果的Tensor。

### 图表
在run_training()这个函数的一开始，是一个Python语言中的with命令，这个命令表明所有已经构建的操作都要与默认的tf.Graph全局实例关联起来。

```
with tf.Graph().as_default():
```
tf.Graph实例是一系列可以作为整体执行的操作。TensorFlow的大部分场景只需要依赖默认图表一个实例即可。

### 会话
完成全部的构建准备、生成全部所需的操作之后，我们就可以创建一个tf.Session，用于运行图表。
```
sess = tf.Session()
```
另外，也可以利用with代码块生成Session，限制作用域：
```
with tf.Session() as sess:
```
Session函数中没有传入参数，表明该代码将会依附于（如果还没有创建会话，则会创建新的会话）默认的本地会话。

生成会话之后，所有tf.Variable实例都会立即通过调用各自初始化操作中的sess.run()函数进行初始化。

```
init = tf.initialize_all_variables()
sess.run(init)
```

sess.run()方法将会运行图表中与作为参数传入的操作相对应的完整子集。在初次调用时，init操作只包含了变量初始化程序tf.group。图表的其他部分不会在这里，而是在下面的训练循环运行。

### 训练循环
完成会话中变量的初始化之后，就可以开始训练了。
训练的每一步都是通过用户代码控制，而能实现有效训练的最简单循环就是：

```

for step in xrange(max_steps):
    sess.run(train_op)
```