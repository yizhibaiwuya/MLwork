# MLwork

## 1. 代码组成部分
### 1.1  configuration
    config.py用于设置模型参数，实现的方法需要其他参数在此处添加。

### 1.2  backbone
    提供resnet18, resnet32两种模型   (resnet32之后会补充)

### 1.3  logs
    用于存储训练日志，便于查看中间结果，！！！保存好最终要提交的模型训练日志

### 1.4  methods!!!
    用于存放实现的方法

## 2. 要添加的代码
    methods/finetune.py中的Finetune类实现了一个最基本的replay式模型。  主要函数包括before_task,  train,  evaluation,  after_task。

    before_task:  负责在每个任务训练前，负责更换分类头，重置优化器，以及一些变量的属性。
    train:  负责利用train loader进行训练。
    evaluation:  负责利用test loader测试结果
    afer_task: 负责在任务结束后更新保存的样本

    要做的：
    需要在methods文件夹下添加一个文件用来实现你的方法，比如methods/icarl.py。  该文件实现了一个类如ICARL，继承了methods/finetune.py文件中的Finetune类。  然后主要实现以上四个函数，如果与Finetune类的相同可以直接复用。

## 3. 需要设置的参数
    在configuration/config.py中:
    数据集dataset: CIFAR10  or CIFAR100
    数据集所在路径dir：根据自己情况设置，  例如数据集在/home/xyk/CIFAR100，  dir = “/home/xyk"
    保存样本数量memory_size:
    模型backbone： resnet12 or resnet18
    优化器相关: opt_name, sched_name
    训练参数相关：n_epoch, batch_size, lr


    在main.py中：
    line: 24  ---> save_path，设置保存训练日志的路径， 最终会存放在 ./logs/DATASET/save_path路径下。
    line: 70  ---> class_of_task，为每个任务划分多少类别， 如为cifar100 划分为[20, 20, 20, 20, 20], 保证总和不超过总的类别数(cifar10-->10,  cifar100-->100), 每个任务的类别数大于0
    line: 78  ---> method，实例化你实现的方法

## 4. 数据集下载链接
    CIFAR10:
    https://drive.google.com/file/d/1NwHL_yBFXiHHp-lFwQZlMmR5ekxmjd4Q/view?usp=sharing
    CIFAR100:
    https://drive.google.com/file/d/1xABKaQlqIvBncK94baXLC2fo-8C5m366/view?usp=sharing

## 5. 关于代码
    需要的backbone或者优化器框架中没有实现，可以考虑自己实现或者用框架中提供的代替。 （如修改了框架请在最终实验报告中说明）

    要实现的方法里的一些功能，可能框架现有的代码不能实现，因此可以适当的修改框架代码。

    部分要实现的论文可能没有官方代码或者官方代码不是pytorch的，可以参考github上别人的实现。

    部分论文的setting可能比较不同，请按照“每个任务类别不相交 + class incremental”来实现。

## 6. 关于精度
    由于数据增广，训练参数等等细节上的不同，无法达到论文报告的精度是正常的。
