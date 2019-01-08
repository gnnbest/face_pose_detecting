
主程序：python codes/main_pred.py

环境：python2.7
      torch : 0.4.1
      torchvision: 0.2.1

功能：检测人脸朝向

步骤：

    用mtcnn检测人脸位置；
    人脸图像经过一定的预处理送给人脸朝向模型进行预测；
    把人脸朝向用立方体画出


结构：

     codes: 存放代码
     models:存放mtcnn检测模型和人脸朝向模型
     pics:存放待检测图像
     output:存放画有人脸方向的图像结果

