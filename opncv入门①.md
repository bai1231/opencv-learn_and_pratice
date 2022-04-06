# opncv

官方社区 https://www.opencv.org/

github官方主页  https://github.com/opencv/opencv



opencv     opencv_contrib

给python里面安装包 

> 先进入对应目录下
>
> C:\> cd /d C:\program\Python\python36  进入目录中
>
> C:\program\Python\python36>python -m pip install opencv-python
>
> 

 

主要研究：

①图像处理与分析

②机器学习与深度学习

③视频分析与追踪

④对象识别与检测



![image-20220329162059498](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220329162059498.png)





## 第一课：读写

![image-20220329163140186](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220329163140186.png)



在github上(https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg)处下载lena的图片

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220329171739367.png #size" alt="image-20220329171739367" style="zoom: 25%;" />

```
 import cv2 as cv
#src=cv.imread("../3h/picture/lena.jpg")
src=cv.imread("../3h/picture/lena.jpg",cv.IMREAD_GRAYSCALE)  #转为灰色图片
print(src)
cv.namedWindow("input",cv.WINDOW_AUTOSIZE)  #自动适应其大小
cv.imshow("input",src)  #在刚刚创建的nameWindow上显示该图片
cv.waitKey(0)  #窗口等待时间
cv.destroyAllWindows() #将创建的所有窗口销毁

```



API知识点：

imread

imshow

nameWindow



## 第二节： 图像创建

Numpy 

创建与初始化图像

API知识



``` 
```

> > Numpy对象
>
> Every image is numpy object in opencv python
>
> numpy包
>
> 常见函数
>
> <img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330081620875.png" alt="image-20220330081620875" style="zoom:50%;" />

\# 默认为浮点数 x = np.zeros(5)   #[0. 0. 0. 0. 0.]

\# 设置类型为整数 y = np.zeros((5), dtype = np.int)  

### 创建

创建一个 ndarray 只需调用 NumPy 的 array 函数

![image-20220330082631230](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330082631230.png)

NumPy 从已有的数组创建数组

![image-20220330082803226](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330082803226.png)

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330082816136.png" alt="image-20220330082816136" style="zoom:50%;" />

```
数值范围创建数组
numpy.arange(start, stop, step, dtype)
x = np.arange(5)   #[0  1  2  3  4]
```

可以和python 的list一样切片



```
NumPy 高级索引
以下实例获取数组中(0,0)，(1,1)和(2,0)位置处的元素。
x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
y = x[[0,1,2],  [0,1,0]]  
print (y)  #[1  4  5]
```



API 知识点：

np.zeros

np.ones

np.random

np.ndarray.reshape



## 第三节：图形绘制

目标：

图形绘制：直线、矩形、圆、椭圆、多边形、文本

填充与描边      生成随颜色

API知识

```

def draw_graphics_demo():
    src=np.zeros((500,500,3),dtype=np.uint8) #type

    #绘制直线
    cv.line(src,(10,10),(400,400),(255,0,0),1,cv.LINE_8,0) #LINE_4 反锯齿填充
    cv.line(src,(400,10),(10,400),(0,255,0),1,cv.LINE_8,0)


    #绘制长方形BGR
    cv.rectangle(src,(100,100),(400,400),(0,0,255),2,cv.LINE_4,0)
    #绘制圆
    cv.circle(src,(250,250),150,(0,255,0),cv.LINE_8,0)
    #绘制椭圆
    cv.ellipse(src,(250,250),(150,50),360,0,360,(255,0,0),3,cv.LINE_4)
    #绘制文本
    cv.putText(src,"Hello opencv",(250,250),cv.FONT_HERSHEY_PLAIN,1.2,(255,0,0),2,cv.LINE_4)
    cv.imshow("input", src)


    #绘制多边形
    src2 = np.zeros((500, 500, 3), dtype=np.uint8)  # type
    points=[]
    points.append((100,100))
    points.append((150, 50))
    points.append((200, 100))
    points.append((200, 300))
    points.append((100, 300))
    for  index in range(len(points)):
        cv.line(src2,points[index],points[(index+1)%5],(0,255,255),2,cv.LINE_4,0)


    #填充矩形
    cv.rectangle(src2, (100, 100), (400, 400), (0, 0, 255), -1, cv.LINE_4, 0) #把线宽改为-1
    cv.imshow("src2", src2)
```



补充知识 

```
np.random.rand() --> 生成指定维度的的[0,1)范围之间的随机数
>>>np.random.rand(4,3,2）
生成一个shape为[4,3,2]的array，array中每个元素都是一个[0,1)之间的随机数

np.random.randn() --> 生成指定维度的服从标准正态分布的随机数，输入参数为维度



>>>np.random.rand(2,2)  等价于 np.random.random(size = (2,2))
也就是说二者都只提供size参数，但一个是位置参数，一个是关键字参数，二者返回的都是[0,1）范围的随机浮点数
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330142054339.png" alt="image-20220330142054339" style="zoom:150%;" />



## 第四节：鼠标响应事件

鼠标事件介绍

监听与响应

API知识点

```
鼠标事件介绍
EVENT_MOUSEMOVE 0            #滑动
EVENT_LBUTTONDOWN 1          #左键点击
EVENT_RBUTTONDOWN 2          #右键点击
EVENT_MBUTTONDOWN 3          #中键点击
EVENT_LBUTTONUP 4            #左键放开
EVENT_RBUTTONUP 5            #右键放开
EVENT_MBUTTONUP 6            #中键放开
EVENT_LBUTTONDBLCLK 7        #左键双击
EVENT_RBUTTONDBLCLK 8        #右键双击
EVENT_MBUTTONDBLCLK 9        #中键双击

def my_mouse_callback(event,x,y,flag,params):  #响应事件， 当前鼠标位置x,y， flag ,用户要传数据
    if event==cv.EVENT_FLAG_LBUTTON:   #如果双击鼠标
        b,g,r=np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)
        cv.circle(params,(x,y),150,(b,g,r),2,cv.LINE_4,0)

def mouse_demo():
    src=np.zeros((512,512,3),dtype=np.uint8)
    cv.namedWindow("mouse_demo",cv.WINDOW_AUTOSIZE)    #namedwondow 和setMouseCallback一定要是同一个窗口，不然会显示没有该窗口
    cv.setMouseCallback("mouse_demo",my_mouse_callback,src) #设置鼠标监听事件   #window_name  事件函数   param
    while True:
        cv.imshow("mouse_demo",src)
        c=cv.waitKey(20)
        if c==27:
            break  #ESC


```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330162948416.png" alt="image-20220330162948416" style="zoom:50%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330163011891.png" alt="image-20220330163011891" style="zoom: 50%;" />



## 第五节：滑块

滑块

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330164056140.png" alt="image-20220330164056140" style="zoom: 50%;align:'left'" />

滑块的创建与响应函数

createTrackbar  #创建

getTrackbarPos  #获得数值





```
src=np.zeros((512,512,3),dtype=np.uint8)  创建512*512*3 其中全为零 
src[:]=[0,0,255]  #把全部数组改成 [0 0  255]
cv.imshow("tb_demo",src)  #就一个窗口
```



```
def do_nothing(p1):
    print(p1) #callback函数里面传的值，就是当前track_bar的值。
    pass
def  track_bar_demo():
    src=np.zeros((512,512,3),dtype=np.uint8)
    cv.namedWindow("tb_demo",cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("B","tb_demo",0,255,do_nothing)       #tracebar_name  window_name  value取值    count最大值     onchange是callback的方法
    cv.createTrackbar("G", "tb_demo", 0,255, do_nothing)
    cv.createTrackbar("R", "tb_demo", 0,255, do_nothing)
    while True:
        b=cv.getTrackbarPos("B","tb_demo")  #tracebar_name  window_name
        g=cv.getTrackbarPos("G","tb_demo")
        r=cv.getTrackbarPos("R","tb_demo")
        src[:]=[b,g,r]
        cv.imshow("tb_demo",src)
        c=cv.waitKey(15)
        if c==27:
            break #ESC
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330171446945.png" alt="image-20220330171446945" style="zoom:33%;" />





## 第六节：像素读写（pixel）

计算机认识的图像

图像像素读写：

API知识点



计算机认识的图像就是三维数组：第一个宽度，第二个高度，第三个（B G R 三个像素通道）

```
print(src.shape)     #(512,512,3)
```



图像基本属性：

> 图像宽
>
> 图像高
>
> 图像色彩空间
>
> 通道数目 #3 BGR
>
> 像素值 ：每个点的BGR值

![image-20220330173050135](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330173050135.png)

API知识点

>shape
>
>src[row,col] 获取图像该行列的像素值





如果是灰度图像，通道数是1 ，shape 就不显示通道数了

```
src = cv.imread("../3h/picture/lena.jpg", cv.IMREAD_GRAYSCALE)
print(src.shape)    #(512, 512)
print(src) 
#[[163 162 161 ... 170 154 130]
 [162 162 162 ... 173 155 126]
 [162 162 163 ... 170 155 128]
 ...
 [ 43  42  51 ... 103 101  99]
 [ 41  42  55 ... 103 105 106]
 [ 42  44  57 ... 102 106 109]]
```





图像读写

>按行列遍历每个像素值
>
>直接读取某个像素值
>
>修改像素值



```
def pixel_demo():
    #BGR
    src =cv.imread("../3h/picture/lena.jpg")
    h,w,ch=src.shape
    print("高：%d  宽： %d  通道数：%d"%(h,w,ch))
    cv.imshow("input",src)
    print("某一个像素点：",src[100,100])

    #改变图像像素点：  #对每个像素点取反，用255减去该值
    for row in range(h):
        for col in range(w):
            b,g,r=src[row,col]
            b=255-b
            g=255-g
            r=255-r
            src[row,col]=[b,g,r]
    cv.imshow("output",src)
    cv.imwrite("C:/Users/LENOVO/Desktop/opencv-picture/Lena_BGR.png",src)
```



## 第七节：单通道和多通道

图像如何表示黑白与彩色

单通道与多通道

API知识点



>单通道就是像素是一个0-255，如灰色图像
>
>多通道的每个像素是三个BGR 每个由0-255

```
src=cv.imread("../3h/picture/lena.jpg",cv.IMREAD_GRAYSCALE)
print(src.shape)       #(512, 512)
src1=cv.imread("../3h/picture/lena.jpg")
print(src1.shape)      # (512, 512, 3)
```

![image-20220330201725540](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330201725540.png)

当图像很大时，可以对图像进行分通道处理，然后合并处理。

```
def chanenels_demo():
    src=cv.imread("../3h/picture/lena.jpg")
    cv.imshow("input",src)
    bgr=cv.split(src)  #把多通道图像分割，分割成单通道的
    cv.imshow("blue",bgr[0])
    cv.imshow("green", bgr[1])
    cv.imshow("red", bgr[2])
    h,w,ch=src.shape

    #对每一个进行改变
    for row in range(h):
        for col in range(w):
            b=bgr[0][row,col]
            g = bgr[1][row, col]
            r = bgr[2][row, col]
            b=255-b
            g=255-g
            r=255-r
            bgr[0][row,col]=b
            bgr[1][row,col]=g
            bgr[2][row,col]=r
    dst=cv.merge(bgr)
    cv.imshow("output",dst)
    cv.imwrite("C:/Users/LENOVO/Desktop/opencv-picture/Lena_BGR1.png",dst)
```



API：
bgr=cv.split(src)  #将彩色多通道图像进行分割为三个通道图像      如果是单通道图像，则会报错

dst=cv.merge(bgr) #将三个单通道的图片进行合并。合并为一个多通道的彩色图片。





## 第八节：镜像翻转

镜像翻转：完全对称的图像





图像坐标

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330214422648.png" alt="image-20220330214422648" style="zoom:50%;" />

API 知识点

dst=cv.flip(src,flipcode)

镜像翻转API：cv.flip(scr, flipcode)

```
xs=cv.flip(src,0)  #沿着哪一个轴翻转 # >0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
```

![image-20220330220233536](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220330220233536.png)

```
#镜像反转
def mirror_demo():
    src=cv.imread("./picture/lena.jpg")
    h,w,c=src.shape
    cv.imshow("input",src)
    dst=np.zeros(src.shape,src.dtype)
    for row in range(h):
        for col in range(w):
            b,g,r=src[row,col]
            dst[row,w-1-col]=[b,g,r]   #以纵轴的中心为基础翻转   即镜面
    cv.imshow("y-flip",dst)

    #API演示
    xs=cv.flip(src,)  #沿着哪一个轴翻转 # >0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
    ys=cv.flip(src,1)
    x_ys=cv.flip(src,-1) #x,y轴同时翻转
    cv.imshow("xs",xs)
    cv.imshow("ys",ys)
    cv.imshow("x_ys",x_ys)
```



## 第九节图像旋转

图像旋转

API知识点

![image-20220331075128265](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220331075128265.png)

API：

rotate:只能旋转固定特殊角度90，180，270

getRotationMatrix2D:  获得旋转矩阵

warpAffine:

```
#图像旋转
def rotate_demo():
    src = cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    dst=cv.rotate(src,cv.ROTATE_90_CLOCKWISE)  #顺时针旋转90°
    dst1=cv.rotate(src,cv.ROTATE_180)  #顺时针旋转180°
    dst2=cv.rotate(src,cv.ROTATE_90_COUNTERCLOCKWISE)  #逆时针旋转90度
    cv.imshow("顺90",dst)
    cv.imshow("顺180",dst1)
    cv.imshow("逆时针90",dst2)

    h,w,ch=src.shape
    cy=h//2
    cx=w//2
    #中心位置（cx,cy）
    M=cv.getRotationMatrix2D((cx,cy),45,1)  #旋转矩阵 参数：旋转中心，旋转角度  大小：和原来的比例
    print(M)
    cos=np.abs(M[0,0])
    sin=np.abs(M[0,1])
    nw=np.int(h*sin+w*cos)  #转动后图片的宽和高
    nh=np.int(h*cos+w*sin)
    M[0,2]+=(nw/2)-cx
    M[1,2]+=(nh/2)-cy
    dst3=cv.warpAffine(src,M,(nw,nh))
    cv.imshow("rotate",dst3)
```



## 第10节：图像的插值

图像插值的场景：仿射变换，畸变矫正，透视变换

常见的插值算法:

临界点插值-inter_nearest  ：速度最快

 双线性插值---inter_linear ：

 双立方插值--inter_cubic  :效果最好



<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220331174421359.png" alt="image-20220331174421359" style="zoom:50%;" />

四种常见图像编辑操作：放大缩小，错切，旋转，平移

API知识点

resize()

```
def resize_image():
    src=cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    h,w=src.shape[:2]
    dst=cv.resize(src,(h//2,w//2),interpolation=cv.INTER_NEAREST)  #image ,dsize(大小)，插值方法:线性插值
    dst1 = cv.resize(src, (h // 2, w // 2), interpolation=cv.INTER_CUBIC)  # image ,dsize(大小)，插值方法:双立法插值
    cv.imshow("dst",dst)
    cv.imshow("dst1",dst1)
```





## 第十一节：图像平移

图像平移

平移矩阵

Api知识点

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401070942091.png" alt="image-20220401070942091" style="zoom:50%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401072809537.png" alt="image-20220401072809537" style="zoom:50%;" />

```
def translate_image():
    src=cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    w,h=src.shape[:2]
    #np.float()就是python里面的float
    M=np.float32([[1,0,100],[0,1,100]])   #平移矩阵  
    dst=cv.warpAffine(src,M,(w,h))  #计算
    cv.imshow("result",dst)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401072656198.png" alt="image-20220401072656198" style="zoom: 33%;" />





## 第十一节：图像算术运算

算术运算：加减乘除

图像像素级别运算

API知识点

#h，w，ch,数据类型，必须都相同才能进行加减乘除四则运算,

 #只有图片大小相同才能进行加减乘除四则运算,

dst=add(src1,src2)

substract

multiply

divide

```
def algorithm_demo():
    src= cv.imread("./picture/apple.jpg")
    src1=cv.imread("./picture/building.jpg")
    #只有图片大小相同才能进行加减乘除四则运算
    cv.imshow("src",src)
    cv.imshow("src1.jpg",src1)
    h,w=src1.shape[:2]
    src2=cv.resize(src,(w,h),interpolation=cv.INTER_CUBIC)
    # cv.imshow("src2",src2)
    # dst=cv.add(src1,src2)       #加法
    # dst1=cv.subtract(src1,src2)  #相减
    # dst2=cv.multiply(src1,src2)   #相乘
    # dst3=cv.divide(src1,src2)     #相除
    # cv.imshow("result",dst)
    # cv.imshow("result1",dst1)
    # cv.imshow("result2",dst2)
    # cv.imshow("result3",dst3)

    bgr=cv.split(src2)
    bgr1=cv.split(src1)
    dst=np.zeros([h,w,3],dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b,g,r=bgr[0][i,j],bgr[1][i,j],bgr[2][i,j]
            b1,g1,r1=bgr1[0][i,j],bgr1[1][i,j],bgr1[2][i,j]
            b2=b+b1 if b+b1<255 else 255
            g2 = g + g1 if g + g1 < 255 else 255
            r2 = r + r1 if r + r1 < 255 else 255
            dst[i,j]=[b2,g2,r2]
    cv.imshow("result4",dst)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401084607362.png" alt="image-20220401084607362" style="zoom:50%;" />





## 第十一节：图像的逻辑运算：

逻辑操作：与，或，非，异或

图像像素级别的操作

API知识点

 dst_and=cv.bitwise_and(src1,src2)
    dst_or=cv.bitwise_or(src1,src2)
    dst_not=cv.bitwise_not(src1)  #非操作，就是255-原来值
    dst_xor=cv.bitwise_xor(src1,src2)  #亦或

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401085822603.png" alt="image-20220401085822603" style="zoom:50%;" />

```
def logic_oprator_demo():
    src1=np.zeros((400,400,3),dtype=np.uint8)
    src2=np.zeros((400,400,3),dtype=np.uint8)
    cv.rectangle(src1,(100,100),(300,300),(255,0,255),-1,cv.LINE_4)
    cv.rectangle(src2,(20,20),(220,220),(0,255,0),-1,cv.LINE_8)
    cv.imshow("input1",src1)
    cv.imshow("input2",src2)
    dst_and=cv.bitwise_and(src1,src2)
    dst_or=cv.bitwise_or(src1,src2)
    dst_not=cv.bitwise_not(src1)  #非操作，就是255-原来值
    dst_xor=cv.bitwise_xor(src1,src2)  #亦或
    src3=cv.imread("./picture/lena.jpg")
    print(src3)
    dst_log=cv.log(np.float32(src3))  #取log
    print(dst_log)
    cv.imshow("log_imge",np.uint8(dst_log*100))
    cv.imshow("dst_and",dst_and)
    cv.imshow("dst_or",dst_or)
    cv.imshow("dst_not",dst_not)
    cv.imshow("dst_xor",dst_xor)
```



## 第十二节：图像亮度与对比度调整

亮度对比度

API：cv.addWeighted（）

![image-20220401110514954](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401110514954.png)





亮度与对比度：对比度就是表征他们俩之间差异大小的值

``` 
图像亮度通俗理解便是图像的明暗程度，如果灰度值在[0，255]之间，则 f 值越接近0亮度越低，f 值越接近255亮度越高
调整亮度，就给每个值加上相同的数值
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401095349395.png" alt="image-20220401095349395" style="zoom:33%;" />

调整亮度：给每个值加上相同的数字  自己手动

```
def lightness_contrast_dem0():
    src=cv.imread("./picture/girl.png")
    src=cv.resize(src,[500,500],interpolation=cv.INTER_CUBIC)
    empty=np.zeros(src.shape,src.dtype)
    cv.imshow("input",src)
    empty.fill(40)
    dst=cv.add(src,empty)
    cv.imshow("add_light",dst)
```



<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401100732497.png" alt="image-20220401100732497" style="zoom: 33%;" />

调整对比度        自己手动

```
#对比度： 扩大每个点的颜色差：给每个点乘上相同的数值
contrast = np.zeros(src.shape, dtype=np.uint8)
contrast.fill(5)
dst1=cv.multiply(src,contrast)   #只有大小相同，图像类型相同，才能进行四则运算
cv.imshow("contrast_imge",dst1)
```

调整

**API：cv.addWeighted（）**

**参数1：src1，第一个原数组.**
**参数2：alpha，第一个数组元素权重**

**参数3：src2第二个原数组**
**参数4：beta，第二个数组元素权重**
**参数5：gamma，图1与图2作和后添加的数值。不要太大，不然图片一片白。总和等于255以上就是纯白色了。**

```
res=cv.addWeighted(src,1.2,empty,0, 40)  #对比度，亮度
cv.imshow("result",res)
```



```
def do_nothing():
    pass
#亮度与对比度
def lightness_contrast_dem0():
    src=cv.imread("./picture/girl.png")
    src=cv.resize(src,[500,500],interpolation=cv.INTER_CUBIC)
    cv.imshow("input",src)

    #亮度  ：给每个值加上相同的数字
    empty=np.zeros(src.shape,src.dtype)
    empty.fill(40)
    dst=cv.add(src,empty)
    cv.imshow("add_light",dst)



    #对比度： 扩大每个点的颜色差：给每个点乘上相同的数值
    contrast = np.zeros(src.shape, dtype=np.uint8)
    contrast.fill(5)
    dst1=cv.multiply(src,contrast)   #只有大小相同，图像类型相同，才能进行四则运算
    cv.imshow("contrast_imge",dst1)

    #API调整cv.addWeighted
    # 参数1：src1，第一个原数组.
    # 参数2：alpha，第一个数组元素权重
    # 参数3：src2第二个原数组
    # 参数4：beta，第二个数组元素权重
    # 参数5：gamma，图1与图2作和后添加的数值。
    res=cv.addWeighted(src,1.2,empty,0, 40)  #对比度，亮度
    cv.imshow("result",res)

    #制作tracebar来操纵亮度
    cv.namedWindow("result",cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("constrast","result",0,100,do_nothing) #tracebar_name  window_name  value取值    count最大值     onchange是callback的方法
    cv.createTrackbar("addlight","result",0,100,do_nothing)
    while True:
        contrast1=cv.getTrackbarPos("constrast","result")/50   #得到的浮点数最大为2，最小为0
        addlight1=cv.getTrackbarPos("addlight","result")
        pic=cv.addWeighted(src,contrast1,empty,0,addlight1)
        cv.imshow("result",pic)
        c=cv.waitKey(10)
        if c==27:
            break#ESC
```





## 第十三节：色彩空间

图像色彩空间介绍

色彩空间转换

根据色彩对象提取

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401163027061.png" alt="image-20220401163027061" style="zoom:33%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401163047070.png" alt="image-20220401163047070" style="zoom: 50%;" />

API知识点

> 色彩空间介绍:
>
> RGB是设备无关的颜色



mask = cv2.inRange(hsv, lower_red, upper_red)

> [OpenCV](https://so.csdn.net/so/search?q=OpenCV&spm=1001.2101.3001.7020)中的inRange()函数可实现二值化功能,更关键的是可以同时针对多通道进行操作，主要是将在两个阈值内的像素值设置为白色（255），而不在阈值区间内的像素值设置为黑色（0）
>
>第一个参数：hsv指的是原图
>
>第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0
>
>第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0
>
>而在lower_red～upper_red之间的值变成255



result=add(src1, src2, dst=None, mask=None, dtype=None)

>参数说明：
> src1, src2：需要相加的两副大小和通道数相等的图像或一副图像和一个标量（标量即单一的数值）
> dst：可选参数，输出结果保存的变量，默认值为None，如果为非None，输出图像保存到dst对应实参中，其大小和通道数与输入图像相同，图像的深度（即图像像素的位数）由dtype参数或输入图像确认
> **mask：图像掩膜，可选参数，为8位单通道的灰度图像，用于指定要更改的输出图像数组的元素，即输出图像像素只有mask对应位置元素不为0的部分才输出，否则该位置像素的所有通道分量都设置为0**
> dtype：可选参数，输出图像数组的深度，即图像单个像素值的位数（如RGB用三个字节表示，则为24位）。
> 返回值：相加的结果图像



```
##第十三节：颜色色彩空间
def color_space_demo():
    src=cv.imread("./picture/green.png")
    # cv.namedWindow("input",cv.WINDOW_AUTOSIZE)
    # cv.imshow("input",src)
    hsv=cv.cvtColor(src,cv.COLOR_BGR2HSV) #BGR转换为HSV色彩空间
    # gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)  #BRG转换为灰度空间
    # YCrcb=cv.cvtColor(src,cv.COLOR_BGR2YCrCb)  #BRG转换为YCrcb色彩空间
    # cv.imshow("HSV",hsv)
    # cv.imshow("gray",gray)
    # cv.imshow("YCrcb",YCrcb)

    #调整HSV色彩空间饱和度
    #实现二值化功能，将位于范围内的变为白色255，不在范围内的变为黑色0
    mask=cv.inRange(hsv,(35,43,46),(77,255,255))  #参数为hsv最小值()   hsv()最大值  #现在绿色的为255，其它为0
    cv.imshow("mask",mask)
    dst=np.zeros(src.shape,src.dtype)
    cv.imshow("dst",dst)
    mask=cv.bitwise_not(mask)  #进行取反 现在非绿为255，绿色为0
    result=cv.add(src,dst,mask=mask)
    print(result)
    cv.imshow("result",result)
```





## 第十四节：浮点数图像（一定要转为整数显示）

像素数据类型

整型与浮点数图像转换

浮点数图像应用

API知识点：convertScaleAbs()  np.uint8(),  astype, dtype



1图像像素数据类型

>np.uint8       cv.imread()读进来就是这种
>
>np.int
>
>np.int32
>
>np.float32
>
>np.double

 #浮点数一定要转换为整数进行显示，否则会进行截断，就出现一些错误,变得很大。所以应先使用 np.uint8()  或者，cv.convertScaleAbs()进行转换为uint8类型。



cv.convertScaleAbs(f_src,alpha=2,beta=100)  #f_src*alpha+beta  把数据类型都转化成CV_8U

```
void cv::convertScaleAbs(
    cv::InputArray src, // 输入数组
    cv::OutputArray dst, // 输出数组
    double alpha = 1.0, // 乘数因子
    double beta = 0.0 // 偏移量
);
```



```
def float_image_demo():
    src=cv.imread("./picture/girl.png")
    h,w=src.shape[:2]
    f_src=np.float32(src)
    f_src=f_src-100
    dst=cv.convertScaleAbs(f_src,alpha=255,beta=0)  ##f_src*alpha+beta  把数据类型都转化成CV_U8
    cv.imshow("input",f_src)
    cv.imshow("dst",dst)

    cv.imshow("dst2",f_src.astype(np.uint8)) #把浮点数类型转换为np.uint8
    
    
    
    src2=np.zeros((400,400,3),dtype=np.double)
src2.fill(100)
cv.imshow("src2",src2)
cv.imshow("src2_int",np.uint8(src2))  #浮点数一定要转换为整数进行显示，否则会进行截断，就出现一些错误。
```

![image-20220401180236357](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401180236357.png)



## 第十五节：图像像素统计及绘制曲线图

图像像素统计信息

应用统计信息

API知识点:cv.mean（）求平均值     cv.meanStdDev（）方差，平均值

绘制曲线图

```
plt.plot(hist,color="r")   #数据  颜色
plt.xlim([0,256])  #x轴大小
plt.show()
```

1、  图像像素统计信息

>像素最大与最小值
>
>均值
>
>方差
>
>灰度分布

2、应用统计信息

>使用均值实现灰度图像分割
>
>使用方差判别空白图像

```
def statistics_demo():
    src = cv.imread("./picture/girl.png")
    h, w = src.shape[:2]
    cv.imshow("input",src)
    mbgr=cv.mean(src)
    mbgr,devbgr=cv.meanStdDev(src)  #每个通道的均值  和   方差
    print("blue mean %d    green mean：%d    red mean: %d"%(mbgr[0],mbgr[1],mbgr[2]))
    print("blue dev %d    green dev：%d    red dev: %d" % (devbgr[0], devbgr[1], devbgr[2]))

    print("min:",np.min(src))
    print("max:",np.max(src))

    #用一个颜色相同的图片测试均值和方差 ：均值：125 方差0
    src2=np.zeros(src.shape,src.dtype)
    src2.fill(125)
    cv.imshow("src2",src2)
    mbgr2,devbgr2 = cv.meanStdDev(src2)  # 每个通道的均值  和   方差
    print("blue mean %d    green mean：%d    red mean: %d" % (mbgr2[0], mbgr2[1], mbgr2[2]))
    print("blue dev %d    green dev：%d    red dev: %d" % (devbgr2[0], devbgr2[1], devbgr2[2]))


    #求像素值
    gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    cv.imshow("gray",gray)
    hist=np.zeros([256],dtype=np.int32)  #存放每个像素值的个数
    h,w=gray.shape[:2]
    for row in range(h):
        for col in range(w):
            pi=gray[row,col]
            hist[pi]+=1

    #利用像素值绘制曲线
    plt.plot(hist,color="r")   #数据  颜色
    plt.xlim([0,256])  #x轴大小
    plt.show()

    #使用均值实现灰度图像分割
    t=cv.mean(gray)[0]
    print(gray.shape)
    binary=np.zeros(gray.shape,gray.dtype)
    print(binary)
    for row in range(h):
        for col in range(w):
            pi=gray[row,col]
            if pi>t:
                binary[row,col]=255
            else:
                binary[row,col]=0
    cv.imshow("binary",binary)
```





## 第十六节：图像查找表

查找表算法

图像查找表映射

API知识点



1、查找表

>LUT-Look Up Table
>
>建立颜色查找表
>
>然后将查找表的宽度为255，然后每个0-255值对应该点的三种色
>
>一一对应。
>
>

2、图像查找表应用：

>灰度图像伪彩色填充
>
>图像像素修正

![image-20220401211033018](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220401211033018.png)



```
def lookup_table_demo():
    image =cv.imread("./picture/rainbow.png")
    cv.imshow("input",image)
    image=cv.resize(image,(256,image.shape[0]),interpolation=cv.INTER_CUBIC)  #先把彩虹颜色图片变为宽256的，然后0-256每一个值，对应彩虹颜色的一个点，形成查找表

    h,w=image.shape[:2]
    print(image.shape)
    #构建查找表
    lut=np.zeros((1,256,3),dtype=np.uint8)
    for col in range(w):
        lut[0,col]=image[h//2,col]
        print(image[h//2,col])
        
        
        
    src=cv.imread("./picture/ellipses.jpg")
    cv.imshow("input",src)
    h,w=src.shape[:2]
    for row in range(h):
        for col in range(w):
            b,g,r=src[row,col]
            b=lut[0,b,0]   #lut表中[1,256,3]  b值对一个下标，该点的绿色通道值，第0个
            g=lut[0,g,1]
            r=lut[0,r,2]
            src[row,col]=(b,g,r)
    cv.imshow("result",src)
```
