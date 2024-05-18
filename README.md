# ExcludeFalseImageStars
exclude false image stars using CPDA and compare with CHT

排除假图像星中的圆拟合算法，使用200张图像比较两种算法的准确率和运行时间

## 组织结构

code文件夹存放的是代码

data文件夹存放你需要转换的图片

## 使用方法

exclude函数中放入你需要转化的图片的相对路径，例如'../data/N1465967021_1.fits'

运行 mian.py 可以将二值化图像保存在image中，

## 算法流程
本算法的实现在 image2circlesCPDA.m 文件中。

预处理 -> 提取骨架 -> 找Y型交点 -> 删除骨架中的Y型交点 -> 用CPDA算法检测V型交点 -> 删除原边缘中的两种交点 -> 合并连通集、拟合圆

取消每一步操作后的注释可以查看中间过程的处理结果
