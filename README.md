# Unet-Family


Unet系列图像语义分割，不断增加新的模型

### 网络结构

**✅**原始的unet网络

**✅**Resnet的残差结构结合Unet

**✅**Attention机制cbam结合Unet

**✅**Mobile net v2 的深度可分离卷积结合Unet

**✅**Shuffle net v2的技巧的结合Unet

### Loss

**✅**BCE-DiceLoss

**✅**Focal Loss

#### NOTE：

Unet系列算法使用设计模式“模板方法”，子类继承自基类UnetBase，子类实现虚函数构建基本函数块。
