# PFLD_106_Landmarks 基于Resnet的PFLD106点人脸关键点检测
在[Hsintao](https://github.com/Hsintao/pfld_106_face_landmarks)大佬代码的基础上进行了修改
## 图片检测demo
<img src="samples.png" ><br> (经过该算法前，事先经过了人脸区域检测)
## 测试结果
  与[Hsintao](https://github.com/Hsintao/pfld_106_face_landmarks)大佬公布的结果做对比
  |     Backbone      |  nme  | 
  | :---------------: | :---: |
  |    MobileNetV2    | 4.96% |  
  |    MobileNetV3    | 4.40% |
  |    Resnet50       | 4.03% |
  
## 数据集地址
+ 数据集来源于飞桨社区
>- 链接：https://aistudio.baidu.com/aistudio/datasetdetail/55093/0

## 数据集准备

  ```bash
  # 下载数据集到data/imgs下
  cd data
  python prepare.py
  ```
  ```bash
  # data 文件夹结构
  data/
    imgs/
    train_data/
      imgs/
      list.txt
    test_data/
      imgs/
      list.txt
  ```
## 开始训练
  ```
  cd ..
  python train.py
  ```
## Reference
https://github.com/Hsintao/pfld_106_face_landmarks
