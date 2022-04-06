# Opencv Python图像处理进阶教程

## 概述：

1、 图像卷积与应用

>图像去噪
>
>图像锐化
>
>边缘发现
>
>图像增强

2、图像直方图

>直方图均衡化
>
>图像直方图比较
>
>直方图反向投影

3、金字塔与模板匹配

>简单的模板匹配
>
>多尺度模板匹配
>
>案例实操



书籍推荐：opencv python3

一勤天下无难事



## 第一节：模糊与卷积原理  均值模糊

图像模糊：Blur

基本原理

API知识：



卷积：加权的滑动平均是一种卷积 。         均值模糊

>一维卷积，就是每个数分别乘以卷积核内的数，然后除以个数，得到的平均值，代替原来的数值
>
>二维卷子：计算该数相邻位置的平均值，代替该数

滤波器=操作数=filter(M)



术语

>卷积核
>
>卷积操作:卷积核的移动，并计算该点值
>
>滤波器：卷积核
>
>滤波 --来自数字信号行业
>
>API-blur

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402104724145.png" alt="image-20220402104724145" style="zoom:33%;" />

API：blur(src,ksize)  ksize:卷积核大小

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402104753140.png" alt="image-20220402104753140" style="zoom:33%;" />



```
#第一节：图像模糊
def blur_demo():
    src=cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    dst=cv.blur(src,(10,10))  #图像，ksize(卷积核大小)， 卷积核越大，所取越多数的平均值，模糊程度越厉害
    dst1 = cv.blur(src, (30, 1))
    cv.imshow("blur image",dst)
    cv.imshow("blur1 image", dst1)
```

![image-20220402091005173](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402091005173.png)



## 第二节：均值与高斯模糊

均值模糊

高斯模糊

API知识：Gaussianblur   blur

>均值模糊：卷积核系数相同，  作用：减少噪声
>
>高斯模糊：卷积核系数不同，  作用：减低噪声

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402104813205.png" alt="image-20220402104813205" style="zoom:33%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402104828416.png" alt="image-20220402104828416" style="zoom:33%;" />

cv.GaussianBlur(src,(5,5),0)   #src,ksize,sigmaX 

 ksize和sigmaX只需要添一个即可，它俩会自动去根据公式计算转化

 当前面ksize设置了，后面的sigmaX就不起作用了

```
def gaussian_blur_demo():
    src = cv.imread("./picture/girl.png")
    cv.imshow("input", src)
    dst=cv.GaussianBlur(src,(5,5),0)   #src,ksize,sigmaX   ksize和sigmaX只需要添一个即可，它俩会自动去根据公式计算转化
    cv.imshow("guassian_blur",dst)
    dst1=cv.GaussianBlur(src,(0,0),10)
    cv.imshow("gassian_blur1",dst1)
    #当前面ksize设置了，后面的sigmaX就不起作用了
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402092834421.png" alt="image-20220402092834421" style="zoom: 33%;" />





## 第三节：统计滤波器

统计滤波器介绍

常见的统计滤波

API知识点



1、统计滤波器介绍：

>最大值滤波          dilate
>
>最小值滤波：用最小值代替该值      erode  
>
>中值滤波：选择滤波器中中间值代替该值        可以用来降低噪声：如椒盐噪声

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402104858094.png" alt="image-20220402104858094" style="zoom:33%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402104918009.png" alt="image-20220402104918009" style="zoom:33%;" />

API知识点

>中值滤波 medianBlur ,  ksize必须大于1的奇数  
>
>erode    :最小值滤波       必须先定义卷积核大小
>
>dilate：最大值滤波

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402100334543.png" alt="image-20220402100334543" style="zoom:50%;" />



中值滤波：可以用来降低噪声

![image-20220402101417407](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402101417407.png)

```
#第三节：统计滤波器
def statictics_filters():
    src = cv.imread("./picture/cat.jpg")
    src=cv.resize(src,(320,400),interpolation=cv.INTER_CUBIC)
    cv.imshow("input", src)


    #最小值滤波
    kernel=np.ones((3,3),np.uint8)  #3*3的卷积核
    dst=cv.erode(src,kernel)
    cv.imshow("minimum_filter",dst)

    #最大值滤波
    kernel=np.ones((3,3),np.uint8)  #3*3的卷积核
    dst2=cv.dilate(src,kernel)
    cv.imshow("maxmum_filter",dst2)

    #中值滤波：可以用来降低噪声如：椒盐噪声
    src1=cv.imread("./picture/girl2_noise.jpg")
    cv.imshow("input1", src1)
    dst3=cv.medianBlur(src1,3)  #这里ksize：仅一个integer  且必须是奇数>=3
    cv.imshow("midian_filtr",dst3)
```







## 第四节：图像添加噪声，去噪

噪声种类：椒盐噪声、高斯噪声

去噪方法：均值滤波，中值滤波，高斯滤波

新去噪方法：非局部均值去噪声

>非局部均值去噪声  --灰度+彩色
>
>匹配窗口和搜索窗口
>
>h参数 ——值越大去噪越厉害，细节丢失

知识点总结

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402160220731.png" alt="image-20220402160220731" style="zoom:33%;" />





添加噪声

```
#添加噪声：
def add_noise():
    src=cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    #添加噪声
    h,w=src.shape[:2]   #高和宽
    print(h)
    rows=np.random.randint(0,h,5000,dtype=np.int)
    cols=np.random.randint(0,w,5000,dtype=np.int)
    for i in range(5000):
        if i%2==1:
            src[rows[i],cols[i]]=(255,255,255)
        else:
            src[rows[i],cols[i]]=(0,0,0)
    cv.imshow("salt and pepper image",src)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402111744875.png" alt="image-20220402111744875" style="zoom:33%;" />



```
#添加噪声，高斯噪声
gniose=np.zeros(src1.shape,src1.dtype)
mean=(15,15,15)  #噪声的均值
sigam=(30,30,30)  #噪声的方差
cv.randn(gniose,mean,sigam) #对该图片产生高斯噪声
cv.imshow("ganssian_noise",gniose)
dst1=cv.add(src1,gniose)
cv.imshow("ganssian_image",dst1)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402112617625.png" alt="image-20220402112617625" style="zoom:33%;" />



三种滤波去噪比较：去除椒盐滤波

![image-20220402113401548](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402113401548.png)

```
#均值滤波
result=cv.blur(src,(5,5))
cv.imshow("result1",result)
#高斯滤波
result2=cv.GaussianBlur(src,(5,5),0)
cv.imshow("result2",result2)
#中值滤波
result3=cv.medianBlur(src,5)
cv.imshow("result3",result3)
```





非局部均值去噪声：去高斯噪声

```
该算法使用自然图像中普遍存在的冗余信息来去噪声。与常用的双线性滤波、中值滤波等利用图像局部信息来滤波不同的是，它利用了整幅图像来进行去噪，以图像块为单位在图像中寻找相似区域，再对这些区域求平均，能够比较好地去掉图像中存在的高斯噪声
cv2.fastNlMeansDenoising()-处理单个灰度图像
cv2.fastNlMeansDenoisingColored()-处理彩色图像。
# h参数调节过滤器强度。大的h值可以完美消除噪点，但同时也可以消除图像细节，较小的h值可以保留细节但也可以保留一些噪点
h = 10
# templateWindowSize用于计算权重的模板补丁的像素大小，为奇数，默认7
templateWindowSize = 3
# searchWindowSize窗口的像素大小，用于计算给定像素的加权平均值，为奇数，默认21
searchWindowSize = 21
dst = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)
```

h越大，去噪越厉害，细节丢失越大       模板窗口在搜索窗口内移动，大小比例应该在1：3左右

```
#去除高斯滤波
#使用非局部均值去噪声，效果比上面三种好
result4=cv.fastNlMeansDenoising(dst1,None,15,15,25)
cv.imshow("result4",result4)
result5 = cv.fastNlMeansDenoisingColored(dst1, None, 15, 15,10, 25)
cv.imshow("result4", result4)
```





## 第五节：边缘与锐化

边缘发现

锐化效果

知识点总结

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402160244446.png" alt="image-20220402160244446" style="zoom:50%;" />

1、图像边缘

>图像边缘特征：
>
>图像边缘类型：

```
#使用Scharr算子：对细节更敏感
```

```
def gradient_demo():
    #robert算子
    robert_x=np.array([[1,0],[0,-1]])
    robert_y=np.array([[0,-1],[1,0]])

    #pewitt 算子
    pewitt_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    pewitt_y=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    #sobel算子
    sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1, 2, -1], [0, 0, 0], [1, 2, 1]])

    #Laplacian 算子
    lap_4=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    lap_8=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    src=cv.imread("./picture/lena.jpg")
    gradx=cv.filter2D(src,cv.CV_16S,sobel_x)  #卷积核算子选择rober_x
    grady=cv.filter2D(src,cv.CV_16S,sobel_y)   #cv_16S ：有符号的16位的整型
    gradx=cv.convertScaleAbs(gradx) #将其变为整型
    grady=cv.convertScaleAbs(grady)
    cv.imshow("x-grady",gradx)
    cv.imshow("y-grady",grady)

    #自带sobel算子
    dx=cv.Sobel(src,cv.CV_32F,1,0)
    dy = cv.Sobel(src, cv.CV_32F, 0, 1)
    dx=cv.convertScaleAbs(dx)
    dy=cv.convertScaleAbs(dy)
    cv.imshow("sobel-x", dx)
    cv.imshow("sobel-y", dy)

    #使用Scharr算子：对细节更敏感
    dx=cv.Scharr(src,cv.CV_32F,1,0)
    dy = cv.Scharr(src, cv.CV_32F, 0, 1)
    dx2=cv.convertScaleAbs(dx)
    dy2=cv.convertScaleAbs(dy)
    cv.imshow("Scharr-x", dx2)
    cv.imshow("Scharr-y", dy2)

    #使用Laplacian算子
    edge=cv.filter2D(src,cv.CV_32F,lap_4)
    cv.imshow("lap",edge)

```

cv.filter2D()

使用自定义内核对图像进行卷积。该功能将任意线性滤波器应用于图像

```
dst=cv.filter2D(src, ddepth, kernel）
ddepth,直接把它设成 -1 就没有任何问题
kernel 很显然表示的是卷积核
```







2、Canny边缘提取算法edges = cv.Canny( image, threshold1, threshold2)

```
Canny()边缘检测步骤
Canny 边缘检测分为如下几个步骤：
步骤 1：去噪。噪声会影响边缘检测的准确性，因此首先要将噪声过滤掉。
步骤 2：计算梯度的幅度与方向。
步骤 3：非极大值抑制，即适当地让边缘“变瘦”。
步骤 4：确定边缘。使用双阈值算法确定最终的边缘信息。

根据当前边缘像素的梯度值（指的是梯度幅度，下同）与这两个阈值之间的关系，判断边缘的属性。具体步骤为：
（1）如果当前边缘像素的梯度值大于或等于 maxVal，则将当前边缘像素标记为强边缘。
（2）如果当前边缘像素的梯度值介于 maxVal 与 minVal 之间，则将当前边缘像素标记为虚
边缘（需要保留）。
（3）如果当前边缘像素的梯度值小于或等于 minVal，则抑制当前边缘像素。
在上述过程中，我们得到了虚边缘，需要对其做进一步处理。一般通过判断虚边缘与强边缘是否连接，来确定虚边缘到底属于哪种情况。通常情况下，如果一个虚边缘：
 与强边缘连接，则将该边缘处理为边缘。
 与强边缘无连接，则该边缘为弱边缘，将其抑制。
```

高阈值应为低阈值2倍             阈值越小，说明越多点被认为边缘，越稠密

```
#边缘提取  :Canny边缘提取算法
def edge_demo():
    src = cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    edge=cv.Canny(src,150,300)  #低阈值   高阈值
    dst=cv.bitwise_and(src,src,mask=edge)  #mask 仅显示白色部分，当作模板，这样与出来，就原来的边缘彩色线条
    cv.imshow("edge",dst)
```



3、图像锐化

laplacian锐化：提高图像中某一部位的清晰度或者焦距程度，提高图像对比度，让细节更加明显

USM锐化算法，课后作业

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402155231563.png" alt="image-20220402155231563" style="zoom:33%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402160316538.png" alt="image-20220402160316538" style="zoom:33%;" />

```
#锐化:Laplacian锐化
def sharpen_image():
    #Laplacian 算子
    lap_5=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    lap_9=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    src=cv.imread("./picture/girl.png")
    cv.imshow("input",src)

    des=cv.filter2D(src,cv.CV_8U,lap_5)
    cv.imshow("output",des)
```





## 第六节：边缘保留滤波

边缘保留滤波介绍

>高斯模糊只考虑了权重，只考虑了[像素](https://so.csdn.net/so/search?q=像素&spm=1001.2101.3001.7020)空间的分布，没有考虑像素值和另一个像素值之间差异的问题，如果像素间差异较大的情况下（比如图像的边缘），高斯模糊会进行处理，但是我们不需要处理边缘，要进行的操作就叫做边缘保留滤波（EPF）

边缘保留滤波:高斯双边模糊

知识点总结

bilateraFilter

edgePreservingFilter

stylization

pencilSketch

```cpp
void bilateralFilter( InputArray src, 
                      int d,
                      double sigmaColor, 颜色空间
                      double sigmaSpace,
                      );


第二个参数，int类型的d，表示在过滤过程中每个像素邻域的直径。如果这个值我们设其为非正数，那么OpenCV会从第四个参数sigmaSpace来计算出它来。
第三个参数，double类型的sigmaColor，颜色空间滤波器的sigma值。这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
第四个参数，double类型的sigmaSpace坐标空间中滤波器的sigma值，坐标空间的标注方差。他的数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。

```

```
#第六节：边缘保留滤波
def edge_perserve_demo():
    src = cv.imread("./picture/edge.jpg")
    cv.imshow("input", src)
    dst=cv.bilateralFilter(src,0,100,10) #高斯双边模糊
    cv.imshow("output",dst)

    #edgePreservingFilter
    dst1 = cv.edgePreservingFilter(src,None,cv.NORMCONV_FILTER,60,0.4)
    cv.imshow("output1", dst1)
    dst2=cv.stylization(src,None,100,0.4)
    dst3,dst4=cv.pencilSketch(src,None,None,60,0.08,0.02)
    cv.imshow("output2",dst2)
    cv.imshow("output3", dst3)
    cv.imshow("output4", dst4)
```



## 第七节：模板匹配

模板匹配原理：

```
工作原理：在待检测图像上，从左到右，从上向下计算模板图像与重叠子图像的匹配度，匹配程度越大，两者相同的可能性越大。
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402173429032.png" alt="image-20220402173429032" style="zoom:33%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402173549691.png" alt="image-20220402173549691" style="zoom: 80%;" />

计算方法：

API知识点：

matchTemplate

minMaxLoc

作业--连连看，程序自动消

### result = cv.**matchTemplate**(target,tpl,md)

```
image参数表示待搜索源图像，必须是8位整数或32位浮点。
templ参数表示模板图像，必须不大于源图像并具有相同的数据类型。
method参数表示计算匹配程度的方法。
result参数表示匹配结果图像，必须是单通道32位浮点。如果image的尺寸为W x H，templ的尺寸为w x h，则result的尺寸为(W-w+1)x(H-h+1)。
其中result是模板图像去匹配的区域位置图像
```

```
cv.minMaxLoc
这个矩阵的最小值，最大值，并得到最大值，最小值的索引
print(min_val,max_val,min_indx,max_indx)
1.0 67.0 (0, 0) (1, 1)
得到矩阵a的最小值为1，索引为（0，0），最大值为67.0索引为（1，1）
```

```
def match_template_demo():
    src=cv.imread("./picture/graf3.png")
    tpl=cv.imread("./picture/tpl.jpg")
    cv.imshow("src",src)
    cv.imshow("tpl",tpl)

    result=cv.matchTemplate(src,tpl,cv.TM_CCOEFF_NORMED)
    cv.imshow("result",result)
    minv,maxv,min_loc,max_loc=cv.minMaxLoc(result) #这个矩阵的最小值，最大值，并得到最大值，最小值的索引
    th,tw=tpl.shape[:2]
    cv.rectangle(src,max_loc,(max_loc[0]+tw,max_loc[1]+th),(0,0,255),2,8,0)
    cv.imshow("match-result",src)
```





## 第八节：直方图比较

图像直方图

比较方法

API知识点：compareHist

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402233918942.png" alt="image-20220402233918942" style="zoom: 50%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220402233955191.png" alt="image-20220402233955191" style="zoom:50%;" />

巴氏距离为0-1  差异越大数值越大

```
def histogran_demo():
    src1 = cv.imread("./picture/lena.jpg")
    src2 = cv.imread("./picture/girl.png")
    src3 = cv.imread("./picture/hrq.jpg")
    cv.imshow("input1",src1)
    cv.imshow("input2",src2)
    cv.imshow("input3",src3)

    gniose=np.zeros(src1.shape,src1.dtype)
    mean=(15,15,15)  #噪声的均值
    sigam=(30,30,30)  #噪声的方差
    cv.randn(gniose,mean,sigam) #对该图片产生高斯噪声
    src4 = cv.add(src1, gniose)
    cv.imshow("src4",src4)

    hist1=cv.calcHist([src1],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])#histsize:即图像数据大小
    hist2=cv.calcHist([src2],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])
    hist3=cv.calcHist([src3],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])
    hist4=cv.calcHist([src4],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])

    dist1=cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)  #巴氏比较方法 ：0-1 数值越大，差异越大
    dist2 = cv.compareHist(hist3, hist2, cv.HISTCMP_BHATTACHARYYA)  # 巴氏比较方法
    dist3 = cv.compareHist(hist1, hist4, cv.HISTCMP_BHATTACHARYYA)  # 巴氏比较方法
    dist4 = cv.compareHist(hist1, hist1, cv.HISTCMP_BHATTACHARYYA)  # 巴氏比较方法
    print("图片1和2之间差异：%f", dist1)
    print("图片3和2之间差异：%f", dist2)
    print("图片1和4之间差异：%f", dist3)
    print("图片1和1之间差异：%f", dist4)
```





## 第九节：直方图反向投影

直方图反向投影的概念

>反向投影就是首先计算某一特征的[直方图](https://so.csdn.net/so/search?q=直方图&spm=1001.2101.3001.7020)模型，然后使用模型去寻找图像中存在的特征。更通俗一点，反向投影可以通过颜色直方图来理解，我们检测图像中某个像素点的颜色是否位于直方图中，如果位于则将颜色加亮，通过对图像的检测，得出结果图像，结果图像一定和直方图像匹配。
>
>方图的反向投影是利用直方图模型计算给定图像像素点的特征。反向投影在某一位置的值是源图像在对应位置的像素值的累计。反向投影操作可实现检测输入源图像给定图像的最匹配区域，可用于目标检测。

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403094622054.png" alt="image-20220403094622054" style="zoom:33%;" />

色彩空间选择 ：HSV（差异更明显）

直方图BIN数目

2D直方图空间

API 知识点：

calcHist:计算直方图的

normalize:归一化 0-255之间  

calcBackProject：直方图反向投影

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403094552081.png" alt="image-20220403094552081" style="zoom:33%;" />

>**反向投影的工作原理**
>
>反向投影图中，某一位置（x，y）的像素值 = 原图对应位置（x，y）像素值在原图的总数目。 即若原图中（5，5）位置上像素值为 200，而原图中像素值为 200 的像素点有 500 个，则反向投影图中（5，5）位置上的像素值就设为 500。
>
>具体步骤：
>
>1. 计算图像直方图：统计各像素值（或像素区间）在原图的总数量。
>2. 将直方图数值归一化到 [0,255] 。
>3. 对照直方图，实现反向投影。



img_backPrj = cv.calcBackProject([sample_hsv], [0, 1],tar_hist,[0,180,0,256],1)

```
void cv::calcBackProject    (   const Mat *     images,
        int     nimages,
        const int *     channels,
        InputArray      hist,
        OutputArray     backProject,
        const float **      ranges,
        double      scale = 1,
        bool    uniform = true 
    )
 const Mat* images:输入图像，图像深度必须位CV_8U,CV_16U或CV_32F中的一种，尺寸相同，每一幅图像都可以有任意的通道数
int nimages:输入图像的数量
const int* channels:用于计算反向投影的通道列表，通道数必须与直方图维度相匹配，
InputArray hist:输入的直方图，直方图的bin可以是密集(dense)或稀疏(sparse)
OutputArray backProject:目标反向投影输出图像，是一个单通道图像，与原图像有相同的尺寸和深度
const float ranges**:直方图中每个维度bin的取值范围
double scale=1:可选输出反向投影的比例因子
bool uniform=true:直方图是否均匀分布(uniform)的标识符，有默认值true

```



## 第十节：图像金字塔

图像金字塔

>上一层必须是下一层的二分之一
>
>

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403094646677.png" alt="image-20220403094646677" style="zoom:33%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403094738161.png" alt="image-20220403094738161" style="zoom:33%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403094706008.png" alt="image-20220403094706008" style="zoom:50%;" />

​			  G2层的高斯金字塔  -  由第一层expend的金字塔图像= 拉普拉斯金字塔第二层



高斯分差

API知识点 ：dst=cv.pyrDown(temp)          expand = cv.pyrUp(pyamid[i], dstsize=src.shape[:2])

```
#图像金字塔
#高斯金字塔
def pyramid_demo(image):
    cv.imshow("input",image)
    level =3
    temp=image.copy()
    pyramid_image=[]
    for i in range(level):
        dst=cv.pyrDown(temp)
        pyramid_image.append(dst)
        cv.imshow("pyramid_down_"+str(i),dst)
        temp=dst.copy()
    return pyramid_image


#拉普拉斯金字塔
def laplaian_demo():
    src = cv.imread("./picture/lena.jpg")
    pyamid = pyramid_demo(src)
    level=len(pyamid)
    for i in range(level-1,-1,-1):
        if (i-1)<0:#最后一层与原图进行减
            expand = cv.pyrUp(pyamid[i], dstsize=src.shape[:2])
            lpls=cv.subtract(src,expand)+127
            cv.imshow("lpls_"+str(i),lpls)
        else:
            expand=cv.pyrUp(pyamid[i],dstsize=pyamid[i-1].shape[:2])  #为高斯金字塔的建立expand层
            lpls=cv.subtract(pyamid[i-1],expand)+127
            cv.imshow("lpls_"+str(i),lpls)
```



## 第十一节：多尺度匹配

多尺度匹配

>原始模板匹配缺点：
>
>多尺度模板匹配：高斯金字塔：就是创造多层高斯金字塔图像，然后利用金字塔的每层图像去匹配，防止因为图片大小与目标图像大小不一致而导致的匹配失败。

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403101419783.png" alt="image-20220403101419783" style="zoom:50%;" />

API知识点

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403101433294.png" alt="image-20220403101433294" style="zoom:50%;" />

```
#多尺度模板匹配
def multiple_sample_template_match():
    target = cv.imread("./picture/traffic.jpg")
    tpl = cv.imread("./picture/traffic_tpl.jpg")
    # target = cv.resize(target, (700, 600), cv.INTER_CUBIC)
    cv.imshow("target",target )
    cv.imshow("template",tpl)

    m_tpl=pyramid_demo(tpl)
    m_tpl.append(tpl)
    count=len(m_tpl)
    t = 0.9
    for i in range(count):
        temp =m_tpl[i]
        th, tw = tpl.shape[:2]
        result = cv.matchTemplate(target,temp,cv.TM_CCOEFF_NORMED)
        # cv.imshow("result", result)
        loc=np.where(result>t)  #找出相关性大于0.8的点
        for pt in zip(*loc[::-1]):
            cv.rectangle(target,pt,(pt[0]+tw,pt[1]+th),(0,0,255),2,0,0)
            cv.imshow("get_match",target)
            break;
        else:
            continue
        break;


    # th, tw = tpl.shape[:2]
    # target1=cv.cvtColor(target,cv.COLOR_BGR2HSV)
    # tpl1 = cv.cvtColor(tpl, cv.COLOR_BGR2HSV)
    # result = cv.matchTemplate(target1,tpl1,cv.TM_CCORR_NORMED)
    # cv.imshow("result", result)
    # t=0.95
    # loc=np.where(result>t)  #找出相关性大于0.8的点
    # for pt in zip(*loc[::-1]):
    #     cv.rectangle(target,pt,(pt[0]+tw,pt[1]+th),(0,0,255),2,0,0)
    #     cv.imshow("get_match",target)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403113237752.png" alt="image-20220403113237752" style="zoom:50%;" />





## 第十二节：总结
