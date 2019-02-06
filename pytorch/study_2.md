## from skimage import io,transform
    skimage的用法:
    import  Image
    img = Image.open(path)#打开图片 
    img.getpixel((height, width))#得到(height, width)处的像素值（可能是一个list，3通道）
    img.convert("L")#转灰度图
    size = (64, 64)
    img.resize(size, Image.ANTIALIAS)#改变尺寸
     box = (10, 10, 100, 100)
    img.crop(box)#在img上的box处截图
     img_data = np.array(img)
     for i in xrange(300):
       x = random.randint(0, img_data.shape[0]-1)
       y = random.randint(0, img_data.shape[1]-1)
       img_data[x][y][0] = 255
     img = Image.fromarray(img_data)#加300个噪音,转来转去麻烦可以直接用skimage度图片就不用转了
     img.rotate(90)#图片旋转90度
     img.transpose(Image.FLIP_LEFT_RIGHT)#图片镜像
### skimage:
 from skimage import io,transform
 img_data = io.imread(img_path)
 transform.resize(img_data, (64, 64))#改变尺寸
 transform.rescale(img_data, 0.5)#缩小/放大图片

## from torch.utils.data import Dataset, DataLoader
## from torchvision import transforms, utils

landmarks = landmarks_frame.iloc[n,1:].as_matrix() #将表格转化为矩阵

landmarks = landmarks.astype('float').reshape(-1,2) #astype(为数据类型)转化为数组
# reshape(-1,b) 利用b来算出a的值来取代-1 -> (a,b)

 plt.pause(0.001) # 暂停一下

    show_landmarks(io.imread(os.path.join('faces/',img_name)),landmarks)
     os.path.join()用于路径拼接文件路径。

    tight_layout #会自动调整子图参数，使之填充整个图像区域。这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分

    ax.axis('off')  #关闭所用坐标轴上的标记，、格栅和单位标记

    assert isinstance() # 插入调试断点到程序

    isinstance()  #当我们定义一个class的时候，我们实际上就定义了一种数据类型。我们定义的数据类型和Python自带的数据类型，比如str、list、dict没什么两样：
    判断一个变量是否是某个类型可以用isinstance()判断

    np.random.randint(a,b) # 生成随机数n: a<=n<=b
    
     __call__  #一个类实例也可以变成一个可调用对象，只需要实现一个特殊方法__call__()