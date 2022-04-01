# Unet-Family


Unet系列图像语义分割，不断增加新的模型

### 目前实现

- [x] 基本的unet网络
- [x] Resnet的残差结构结合Unet
- [x] Attention机制cbam结合Unet
- [x] mobile net v2 的深度可分离卷积结合Unet
- [x] Shuffle net v2的技巧的结合Unet

### NOTE：

Unet系列算法使用设计模式“模板方法”，子类继承自基类UnetBase，子类实现虚函数构建基本函数块。
