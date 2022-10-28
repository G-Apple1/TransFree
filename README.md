# TransFree
### 基于自注意力机制与无锚点的仔猪姿态识别
### Recognition of piglet postures based on self-attention mechanism and anchor-free method
Author:
Xu Chengguo, Xue Yueju, Hou Wenhao, Guo jingfeng, Wang Xiarui 

该文结合 Transformer 网络与无锚点目标检测头，提出了一种新的仔猪姿态识别模型 TransFree（ Transformer + Anchor-Free）。该模型使用 Swin Transformer 作为基础网络，提取仔猪图像的局部和全局特征，然后经过一个特征增强模块（ Feature Enhancement Module， FEM）进行多尺度特征融合并得到高分辨率的特征图，最后将融合后的特征图输入 Anchor-Free 检测头进行仔猪的定位和姿态识别。


![image](https://user-images.githubusercontent.com/62458945/198507832-5a85b6e1-4c25-46bb-9b8e-12bf1ef13492.png)

![image](https://user-images.githubusercontent.com/62458945/198508204-2aee1148-0fa9-4ad2-8f5c-e9b7c0086603.png)

