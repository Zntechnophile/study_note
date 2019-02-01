#Pytorch基础
##什么是Pytorch
	Pytorch是一个动态的建图工具。不像tensorflow那样，先建图，然后通过feed和run重复执行建好的图，相对来说，Pytorch具有更好的灵活性。
##编写一个深度神经网络需要关注的地方是：
1. 网络的参数应该由什么对象保存
2. 如何构建网络
3. 如何计算梯度和更新参数

##Torch的简单数学运算
**Torch的数学运算与numpy高度相似**

**使用时，需要将数据转化Tensor类型 tensor = torch.FloatTensor(data)**

	绝对值运算（abs）         torch.abs(tensor)
	三角函数（sin/cos/tan)    torch.sin(tensor)
	均值（mean）              torch.mean(tensor)
	
	矩阵运算            torch.mm(tensor, tensor)
	
##保存参数
***pytorch有两种变量类型，一个是Tensor，一个是Variable***
####Tensor：
就像**ndarray**一样，一维的**Tensor**叫**Vector**，二维的**Tensor**叫**Matrix**，三维及以上叫**Tensor**

####Variable:
是Tensor的一个wrapper（封装），不仅保存了值，还保存了这个值的creator
	torch.Tensor(2,3,4) 
	#创建一个未初始化的变量
	
	torch.add(a,b,out=x) 
	#使用Tensor()方法创建出来的Tensor用来接收计算结果，当然torch.add(...)也会返回计算结果
	
	a.add_(b) 
	#所有带_的operation,都会更改调用对象的值。 例如a=1;b=2;a.add_(b);则a=3; 没有_的operation就没有这种效果，只会返回运算结果。
##Pytorch核心功能
####torch.autograd提供了类和函数用来对任意标量函数进行求导。要想使用自动求导，只需要对已有的代码进行微小的改变。只需要将所有的tensor包含进Variable对象中即可。
	代码如下:
	from torch.autograd import Variable
	x = torch.rand(5)
	x = Variable(x, requaires_grad = true)
	grad = torch.FloatTensor([1,2,3,4,5])
	y.backward(grad)
	#如果y是scalar的话，那么直接y.backward()，然后通过x.grad方式，就可以得到var的梯度如果y不是scalar，那么只能通过传参的方式给x指定梯度
##neural networks
###使用torch.nn包中的工具来构建神经网络的步骤：
1. 定义神经网络的权重，搭建网络结构
2. 遍历整个数据集进行训练
3. 将数据输入到整个神经网络
4. 计算loss
5. 计算网络权重的梯度
6. 更新网络权重 weight = weight + learning_rate + gradient
####示例程序
	import torch.nn as nn
	import torch.nn.functional as F
	class Net(nn.Module):#需要继承这个类
    def __init__(self):
        super(Net, self).__init__()
        #建立了两个卷积层，self.conv1, self.conv2，注意，这些层都是不包含激活函数的
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        #三个全连接层
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
 
    def forward(self, x): #注意，2D卷积层的输入data维数是 batchsize*channel*height*width
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension（除批次）
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
 
	net = Net()
	net  #运行程序
####输出结果
	Net (
	  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
	  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
	  (fc1): Linear (400 -> 120)
	  (fc2): Linear (120 -> 84)
	  (fc3): Linear (84 -> 10)
		)
####测试
	len(list(net.parameters())) 
	>> 10
	#为什么是10呢？ 因为不仅有weights，还有bias， 10=5*2。
    #list(net.parameters())返回的learnable variables 是按照创建的顺序来的
    #list(net.parameters())返回 a list of torch.FloatTensor objects
	input = Variable(torch.randn(1, 1, 32, 32))
	out = net(input)
	#这个地方就神奇了，明明没有定义__call__()函数啊，所以只能猜测是父类实现了，并且里面还调用了forward函数
	out 
	#查看源码之后，果真如此。那么，forward()是必须要声明的了，不然会报错
	out.backward(torch.randn(1, 10))
##训练网络
***torch.nn包下有很多loss的标准，同时torch.optimizer帮助完成更新权重的工作，这样可以手动更新参数***
	learning_rate = 0.01
	for f in net.parameters():
	f.data.sub_(f.grad.data * learning_rate) # 有了optimizer就不用写这些了
	import torch.optim as optim
	# create your optimizer
	optimizer = optim.SGD(net.parameters(), lr = 0.01)
	# in your training loop:
	optimizer.zero_grad() 
	# zero the gradient buffers，如果不写这个函数，也是可以正常工作的，不知这个函数的必要性在哪？
	output = net(input)
	# 这里就体现出来动态建图了，你还可以传入其他的参数来改变网络的结构
	loss = criterion(output, target)
	loss.backward()
	optimizer.step() # Does the update

##其他要注意的问题
**只有定义的Variable才会被求梯度，有creator创造的不会去求梯度，因此需要这步操作Variable(Tensor, requaires_grad=True)**
 


	
	