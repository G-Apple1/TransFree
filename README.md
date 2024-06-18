# TransFree
### 基于自注意力机制与无锚点的仔猪姿态识别
### Recognition of piglet postures based on self-attention mechanism and anchor-free method
Author:
Xu Chengguo, Xue Yueju, Hou Wenhao, Guo jingfeng, Wang Xiarui 

**农业工程学报**: http://www.tcsae.org/nygcxb/article/abstract/20221419?st=search

该文结合 Transformer 网络与无锚点目标检测头，提出了一种新的仔猪姿态识别模型 TransFree（ Transformer + Anchor-Free）。该模型使用 Swin Transformer 作为基础网络，提取仔猪图像的局部和全局特征，然后经过一个特征增强模块（ Feature Enhancement Module， FEM）进行多尺度特征融合并得到高分辨率的特征图，最后将融合后的特征图输入 Anchor-Free 检测头进行仔猪的定位和姿态识别。


![image](https://user-images.githubusercontent.com/62458945/198507832-5a85b6e1-4c25-46bb-9b8e-12bf1ef13492.png)

![image](https://user-images.githubusercontent.com/62458945/198508204-2aee1148-0fa9-4ad2-8f5c-e9b7c0086603.png)


**下载模型权重**
链接：https://pan.baidu.com/s/1xYRWnf70EtzTt8vDt9wYcw 
提取码：Gguo 


**demo命令行**
```python
/home/server/xcg/CenterNet/src/demo.py
ctdet
--dataset
pig_coco
--keep_res
--vis_thresh
0.5
--arch
swint
--load_model
/home/server/xcg/CenterNet/exp/ctdet/ctdet_coco_swint_2/model_last-2.pth
```


**test命令行**
```python
/home/server/xcg/CenterNet/src/test.py
ctdet
--exp_id
test
--soft-nms
--not_prefetch_test
--arch
swint
--keep_res
--load_model
/home/server/xcg/CenterNet/exp/ctdet/ctdet_coco_swint_2/model_last-2.pth
```

**train命令行**
```python
/home/server/xcg/CenterNet/src/main.py
ctdet
--exp_id
ctdet_coco_swint_none
--batch_size
8
--arch
swint-none
--lr
1.45e-4
--lr_step
50,90
--gpus
0
--num_workers
16
--num_epochs
100
--val_intervals
1
--keep_res
--load_model
/home/server/xcg/CenterNet/weights/model_final.pth
```

**运行demo pycharm代码**

![image](https://user-images.githubusercontent.com/62458945/204023214-c9c43738-65ac-445d-8fed-3d0f33c28aef.png)

**运行test pycharm代码**

![image](https://user-images.githubusercontent.com/62458945/204023522-f84dea20-8ca1-4cfa-82f6-5e0724bc62fd.png)

**运行train pycharm代码**

![image](https://user-images.githubusercontent.com/62458945/204023858-dcc9e809-e2d6-46d9-886c-8bf5d78796a7.png)
