 # opencv 图像与视频分析教程

二值图像分析

>图像二值化
>
>二值图像轮廓分析
>
>霍夫检测
>
>图像检测与几何形状识别
>
>轮廓匹配
>
>形态学

视频读写

>视频读写
>
>视频背景分析
>
>颜色对象提取
>
>案例分析

视频内容分析

案例实战



## 第一节：认识二值图像

二值图像的定义与说明

只有0或255俩种值

简单图像二值化：

>手动选取阈值，Trackbar ，无厘头的方式

![image-20220403151532779](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403151532779.png)

API：二值化函数threshold（）

  #二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY

```
def binary_demo():
    src=cv.imread("./picture/lena.jpg",cv.IMREAD_GRAYSCALE) #灰度图像
    cv.imshow("input",src)
    #二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY
    ret,dst=cv.threshold(src,127,255,cv.THRESH_BINARY) #二值化分割    ret:阈值， dst:二值化后的图像
    cv.imshow("binary",dst)
    print(ret)

    #建立Tracebar
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("Threshold", "input", 0, 255,do_nothing)  # tracebar_name  window_name  value取值    count最大值     onchange是callback的方法
    while True:
        threshold=cv.getTrackbarPos("Threshold", "input")
        ret, dst = cv.threshold(src, threshold, 255, cv.THRESH_BINARY)
        cv.imshow("input",dst)
        t=cv.waitKey(10)
        if t==27:
            break#ESC
```



## 第二节：图像二值化

图像二值化：将灰度图像通过阈值实现分割，就是图像的阈值化

图像二值化是阈值分割的二分类分割算法

>THRESH_BINARY:将低于阈值的变为0，高于阈值的变为255
>
>THRESH_BINARY_INV : 与上一个相反，将高于的置为0，低于的置为255
>
>THRESH_TRUNC:截断，将高于阈值的变为255
>
>THRESH_TOZERO:截断，将低于阈值的变为0
>
>THRESH_TOZERO_INV,将高于阈值的变为0

![image-20220403151514561](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403151514561.png)



  #二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY![image-20220403151751730](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403151751730.png)

API：threshold（）





阈值方法共四种：有全局阈值和自适应阈值

## 第三节：全局阈值

全局阈值：

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403154410181.png" alt="image-20220403154410181" style="zoom:50%;" />

①均值法

OTSU阈值：通过寻找内类最小方差来寻去t阈值

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403154453671.png" alt="image-20220403154453671" style="zoom:50%;" />

三角阈值：

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403154510194.png" alt="image-20220403154510194" style="zoom:50%;" />

![image-20220403154725279](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403154725279.png)





```
#第二节：全局阈值
def threshold_segmentation_demo1():
    src=cv.imread("./picture/lena.jpg",cv.IMREAD_GRAYSCALE) #灰度图像
    cv.imshow("input",src)
    #二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY

    ret1, dst1 = cv.threshold(src, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    ret2, dst2= cv.threshold(src, 0, 255, cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
    cv.imshow("threshold_segmentation1", dst1)
    cv.imshow("threshold_segmentation2", dst2)
    print(ret1) #117.0
    print(ret2)#115.0
```



<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403155004711.png" alt="image-20220403155004711" style="zoom:50%;" />





## 第四节：自适应阈值

自适应阈值概述

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403155722644.png" alt="image-20220403155722644" style="zoom:50%;" />

![image-20220403160001153](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403160001153.png)

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403155759469.png" alt="image-20220403155759469" style="zoom:50%;" />

![image-20220403155834928](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403155834928.png)

API：adaptiveThreshold（）

```
void adaptiveThreshold(InputArray src, OutputArray dst,  
2.                           double maxValue, int adaptiveMethod,  
3.                           int thresholdType, int bolckSize, double C)  

参数1：InputArray类型的src，输入图像，填单通道，单8位浮点类型Mat即可。
参数2：函数运算后的结果存放在这。即为输出图像（与输入图像同样的尺寸和类型）。
参数3：预设满足条件的最大值。
参数4：指定自适应阈值算法。可选择ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C两种。（具体见下面的解释）。
参数5：指定阈值类型。可选择THRESH_BINARY或者THRESH_BINARY_INV两种。（即二进制阈值或反二进制阈值）。
参数6：表示邻域块大小，用来计算区域阈值，一般选择为3、5、7......等。
参数7：参数C表示与算法有关的参数，它是一个从均值或加权均值提取的常数，可以是负数。（具体见下面的解释）。

ADAPTIVE_THRESH_MEAN_C，为局部邻域块的平均值，该算法是先求出块中的均值，再减去常数C。

ADAPTIVE_THRESH_GAUSSIAN_C，为局部邻域块的高斯加权和。该算法是在区域中(x, y)周围的像素根据高斯函数按照他们离中心点的距离进行加权计算，再减去常数C。
```





```
def threshold_method_demo():
    src = cv.imread("./picture/text.jpg", cv.IMREAD_GRAYSCALE)  # 灰度图像
    cv.imshow("input", src)
    # 二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY
    dst1 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,10,None)
    dst2 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,25,10)
    cv.imshow("threshold_segmentation1", dst1)
    cv.imshow("threshold_segmentation2", dst2)
```

![image-20220403161650359](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403161650359.png)



opencv共支持四种阈值分割方法：全局阈值：OTSU阈值、三角阈值

​															自适应阈值：







## 第五节：去噪声对二值化的影响

去噪声对全局值域的影响

![image-20220403174547414](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403174547414.png)



<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403174608535.png" alt="image-20220403174608535" style="zoom:50%;" />

API

![image-20220403174624491](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403174624491.png)阈值的几种方法

```
def threshold_noise_demo():
    src = cv.imread("./picture/rice_noise.jpg", cv.IMREAD_GRAYSCALE)  # 灰度图像
    cv.imshow("input", src)
    # src=cv.GaussianBlur(src,(3,3),5)  #高斯去噪
    src = cv.medianBlur(src, 5)  #中值去噪
    # 二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY
    dst1 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,10,None)
    res,dst2 = cv.threshold(src,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)  #OTSU阈值
    cv.imshow("threshold_segmentation1", dst1)
    cv.imshow("threshold_segmentation2", dst2)
```

![image-20220403174202961](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403174202961.png)





## 第六节：连通组件扫描

基本概念解释：

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403175753263.png" alt="image-20220403175753263" style="zoom:50%;" />

基于图搜索算法

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403175724525.png" alt="image-20220403175724525" style="zoom:50%;" />

俩部分法：只看上面和左边，先标记数字，再合并。

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403175815519.png" alt="image-20220403175815519" style="zoom:33%;" />

基于等价队列合并

![image-20220403175934026](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403175934026.png)



```
int cv::connectedComponents (
    cv::InputArrayn image, // input 8-bit single-channel (binary)
    cv::OutputArray labels, // output label map
    int connectivity = 8, // 4- or 8-connected components
    int ltype = CV_32S // Output label type (CV_32S or CV_16U)
);
```

其中 connectedComponents()仅仅创建了一个标记图（图中不同连通域使用不同的标记，和原图宽高一致），

```
#第六节：连通组件扫描
def connected_components_demo():
    src = cv.imread("./picture/rice_noise.jpg", cv.IMREAD_GRAYSCALE)  # 灰度图像
    cv.imshow("input", src)
    src = cv.medianBlur(src, 5)  #中值去噪
    res,dst1 = cv.threshold(src,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)  #OTSU阈值
    cv.imshow("threshold_segmentation2", dst1)

    output=cv.connectedComponents(dst1,ltype=cv.CV_32S)  #ltype输出的数据的类型
    print(output) #算上背景一共26个区域，其中25个白色的非连通区域
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403180921911.png" alt="image-20220403180921911" style="zoom: 50%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403180952260.png" alt="image-20220403180952260" style="zoom:33%;" />

![image-20220403182152450](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403182152450.png)



<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403195031375.png" alt="image-20220403195031375" style="zoom:50%;" />

```
output_more=cv.connectedComponentsWithStats(dst1,connectivity=8,ltype=cv.CV_32S)
num_lables=output_more[0]  #区域数
lables=output_more[1] #标记图
stats=output_more[2]  #每个区域的状态x,y,width,height,area
centers=output_more[3]  #每个区域的中心点，为float类型，在显示时需要转为int类型  np.int(cx)
```



```
output_more=cv.connectedComponentsWithStats(dst1,connectivity=8,ltype=cv.CV_32S)
num_lables=output_more[0]
lables=output_more[1]
stats=output_more[2] #每个区域的状态x,y,width,height,area
centers=output_more[3] #每个区域的中心点
image_stats=np.copy(src)
image_stats=cv.cvtColor(image_stats,cv.COLOR_GRAY2BGR)
for i in range(num_lables):
    if i==0: #背景这一区域
        continue
    cx,cy=centers[i] #中心位置，为float数据
    x,y,width,height,area=stats[i]
    cv.rectangle(image_stats,(x,y),(x+width,y+height),(0,0,255),2,0,0)
    cv.circle(image_stats,(np.int(cx),np.int(cy)),2,(255,0,0),-1,0,0)
cv.imshow("statistic",image_stats)
```



![image-20220403193322982](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403193322982.png)





## 第七节：轮廓发现

基本概念解释：

图像轮廓-图像边界

主要针对二值图像，轮廓是一系列点的集合

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403195053196.png" alt="image-20220403195053196" style="zoom:50%;" />



API知识点：findContours

​					drawContours

![image-20220403195130987](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403195130987.png)

contours, hierarchy = cv.findContours( image, mode, method[, contours[, hierarchy[, offset]]] )

1. 参数1：源图像

2. 参数2：轮廓的检索方式，这篇文章主要讲解这个参数

3. 参数3：一般用 cv.CHAIN_APPROX_SIMPLE，就表示用尽可能少的像素点表示轮廓

4. contours：图像轮廓坐标，是一个链表

5. hierarchy：[Next, Previous, First Child, Parent]，文中有详细解释

   

```
def find_contours_demo():
    src = cv.imread("./picture/coins.jpg")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src,(3,3),0)#高斯去噪
    gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    res,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)  #OTSU阈值
    cv.imshow("threshold_segmentation2", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #只绘制外轮廓  #cv.RETR_TREE内外轮廓都会绘制
    for i in range(len(contours)):
        cv.drawContours(src ,contours, i, (0, 0, 255), 2)
    cv.imshow("contpurs", str)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403203417834.png" alt="image-20220403203417834" style="zoom:50%;" />

```
cv.RETR_EXTERNAL外轮廓
cv.RETR_TREE内外轮廓都会绘制
```

![image-20220403203625799](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403203625799.png)





## 第八节：图像测量：面积，周长

图像测量

>计算面积与周长  单位：像素单位
>
>
>
>外接矩形与横纵比率

API知识点：

cv.boundingRect:外接矩形 最小x,y 和最大x,y计算

cv.contourArea:面积，像素个数

cv.arcLength：弧长

#基于界边界求：

 area=cv.contourArea(contours[i])  #求面积
 arclen=cv.arcLength(contours[i],True)#求周长  closed:是否是封闭
 x,y,w,h=cv.boundingRect(contours[i]) #外接矩形 的x,y,w,h

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403210520083.png" alt="image-20220403210520083" style="zoom:33%;" />

```
#第八节：图像测量：面积，周长
def measure_contours_demo():
	#获得二值图像
    src = cv.imread("./picture/right.jpg")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src, (3, 3), 0)  # 高斯去噪
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    res, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # OTSU阈值
    cv.imshow("threshold_segmentation2", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#求外边界
    for i in range(len(contours)):
        #求面积：
        area=cv.contourArea(contours[i])
        arclen=cv.arcLength(contours[i],True)#求周长  closed:是否是封闭
        x,y,w,h=cv.boundingRect(contours[i]) #外接矩形 的x,y,w,h
        print("area:%d,  arclen: %d"%(area,arclen))
        if area<50 or arclen<20:  #过滤掉面积太小，或者，周长太小的
            continue
        #利用横纵比找，斜度
        # ratio=np.minimum(w,h)/np.maximum(w,h)
        # if ratio>0.8:
        #     continue
        cv.drawContours(src, contours, i, (0, 0, 255), 2)
    cv.imshow("contpurs", src)
```





## 第九讲：几何分析

几何距：

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403233342035.png" alt="image-20220403233342035" style="zoom:50%;" />

故中心位置x=M10/M00   y=M01/M00

计算角度与中心：

contours：外边轮廓发现

API：cv.moments(contour)  :求几何距

​	cv.fitEllipse(contour)

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403233504163.png" alt="image-20220403233504163" style="zoom:50%;" />



```
#第九讲：几何分析
def measure_contours_demo1():
    src = cv.imread("./picture/rice_noise.jpg")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src, (3, 3), 0)  # 高斯去噪
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    res, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # OTSU阈值
    cv.imshow("threshold_segmentation2", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#求外边界
    for i in range(len(contours)):
        #求面积：
        area=cv.contourArea(contours[i])
        arclen=cv.arcLength(contours[i],True)#求周长  closed:是否是封闭
        x,y,w,h=cv.boundingRect(contours[i]) #外接矩形 的x,y,w,h
        print("area:%d,  arclen: %d"%(area,arclen))
        if area<50 or arclen<20:  #过滤掉面积太小，或者，周长太小的
            continue
        # 利用横纵比找，斜度
        ratio=np.minimum(w,h)/np.maximum(w,h)
        if ratio>0.8:
            mm=cv.moments(contours[i])
            m00=mm["m00"]
            m10 = mm["m10"]
            m01 = mm["m01"]
            cx=np.int(m10/m00)
            cy=np.int(m01/m00)
            (x,y),(a,b),degree=cv.fitEllipse(contours[i])  #左上角位置，长短轴，角度
            print("长轴%d 短轴：%d  角度%d"%(b,a,degree))
            cv.circle(src,(cx,cy),2,(255,0,0),-1,8,0)  #绘制中心
            cv.putText(src,str(np.int(degree)),(cx-20,cy-20),cv.FONT_HERSHEY_PLAIN,1.0,(255,0,255),1)
            cv.drawContours(src, contours, i, (0, 0, 255), 2)

    cv.imshow("contpurs", src)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220403233943087.png" alt="image-20220403233943087" style="zoom: 50%;" />



## 第十节：距离变换

>距离变换
>
>距离度量

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404091220895.png" alt="image-20220404091220895" style="zoom:50%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404091257238.png" alt="image-20220404091257238" style="zoom: 50%;" />



<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404091323327.png" alt="image-20220404091323327" style="zoom:50%;" />



API：知识点

```
dist=cv.distanceTransform(binary,cv.DIST_L1,3,dstType=cv.CV_8U)
```

```
dist1 = cv.distanceTransform(binary1, cv.DIST_L2,3, dstType=cv.CV_32F)
```

![image-20220404091019626](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404091019626.png)

```
#第十节：距离变换
def distance_demo():
    src = cv.imread("./picture/right1.jpg")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src, (3, 3), 0)  # 高斯去噪
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    res, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # OTSU阈值
    cv.imshow("threshold_segmentation2", binary)

    #距离变换
    dist=cv.distanceTransform(binary,cv.DIST_L1,3,dstType=cv.CV_8U)
    cv.imshow("distance-transform",dist)






    src1 = cv.imread("./picture/rice_noise.jpg")
    cv.imshow("input", src1)
    src1 = cv.medianBlur(src1, 5)  # 中值去噪
    gray = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
    res, binary1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # OTSU阈值
    cv.imshow("threshold", binary1)

    dist1 = cv.distanceTransform(binary1, cv.DIST_L2,3, dstType=cv.CV_32F)
    dist2=cv.normalize(dist1,0,255,cv.NORM_MINMAX)  #把区间变换到0-255,  此时为浮点数，下面直接show时，会出现截断
    print(dist2)
    res,binary2=cv.threshold(dist2,1.5,255,cv.THRESH_BINARY)
    cv.imshow("distanc_transform2",dist2)
    cv.imshow("binary2",binary2)
```

![image-20220404090913832](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404090913832.png)



## 第十一节：点多边形测试

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404093904146.png" alt="image-20220404093904146" style="zoom:50%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404093841698.png" alt="image-20220404093841698" style="zoom:50%;" />





应用场景：

>判断一个点是否在一个多边形内
>
>判断一个对象是否在指定范围之内

API知识点：

retval=cv.pointPolygonTest(contour,pt,measureDist)

```
 pointPolygonTest(InputArray contour, Point2f pt, bool measureDist)

contour 某一轮廓集合点
 Point2f 某一测试点
用于测试一个点是否在多边形中
当measureDist设置为true时，返回实际距离值。若返回值为正，表示点在多边形内部，返回值为负，表示在多边形外部，返回值为0，表示在多边形上。
当measureDist设置为false时，返回 -1、0、1三个固定值。若返回值为+1，表示点在多边形内部，返回值为-1，表示在多边形外部，返回值为0，表示在多边形上。

```



```
#第十一节：点多边形测试
def point_polygon_test_demo():
    src = cv.imread("./picture/right1.jpg")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src, (3, 3), 0)  # 高斯去噪
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    res, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # OTSU阈值
    cv.imshow("binary", binary)

    contours,hierachy=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)  #轮廓查找
    print("total number",len(contours))

    #不计算距离
    h,w=src.shape[:2]
    for row in range(h):
        for col in range(w):
            dist_flag=cv.pointPolygonTest(contours[0],(col,row),False)  #为False返回 -1、0、1三个固定值。若返回值为+1，表示点在多边形内部
            if dist_flag>0:                                             #当measureDist设置为true时，返回实际距离值。若返回值为正，表示点在多边形内部，返回值为负，表示在多边形外部
                src[row,col]=(255,0,0)
            if dist_flag<0:
                src[row,col]=(0,0,255)
    cv.imshow("ppt-demp",src)

    #计算距离
    h,w=src.shape[:2]
    src1=np.copy(src)
    for row in range(h):
        for col in range(w):
            dist=cv.pointPolygonTest(contours[0],(col,row),True)  #为False返回 -1、0、1三个固定值。若返回值为+1，表示点在多边形内部
            if dist>0:   #内部                                          #当measureDist设置为true时，返回实际距离值。若返回值为正，表示点在多边形内部，返回值为负，表示在多边形外部
                src1[row,col]=(np.abs(dist),0,0)
            if dist<0:   #外部
                src1[row,col]=(0,0,155-np.abs(dist))
    cv.imshow("ppt-demp1",src1)
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404093538646.png" alt="image-20220404093538646" style="zoom:67%;" />





## 第十二节：图像投影

图像投影的基本概念

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404094950265.png" alt="image-20220404094950265" style="zoom:33%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404095009796.png" alt="image-20220404095009796" style="zoom:50%;" />

插值拟合与权重分割

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404095034983.png" alt="image-20220404095034983" style="zoom:50%;" />

API知识点：没有新的API知识点，就是遍历，记录每行每列的点数，用plt显示

二值化

像素统计

直方图显示



```
from matplotlib import pyplot as plt
def binary_projection_demo():
    src = cv.imread("./picture/canjian.jpg")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src, (3, 3), 0)  # 高斯去噪
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    res, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # OTSU阈值
    cv.imshow("binary", binary)

    #投影
    h,w=gray.shape
    y_projection=np.zeros((h),dtype=np.int32)
    x_projection = np.zeros((w), dtype=np.int32)
    for row in range(h):
        count=0
        for col in range(w):
            pv=binary[row,col]
            if pv==255:  #因为二值化过了
                count+=1
        y_projection[row]=count

    for col in range(w):
        count=0
        for row in range(h):
            pv=binary[row,col]
            if pv==255:  #因为二值化过了
                count+=1
        x_projection[col]=count
    plt.plot(y_projection,color="b")
    plt.xlim([0,h])
    plt.show()

    plt.plot(x_projection, color="r")
    plt.xlim([0, w])
    plt.show()
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404100407171.png" alt="image-20220404100407171" style="zoom:33%;" /><img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404100419255.png" alt="image-20220404100419255" style="zoom:33%;" />





## 第十三节：轮廓匹配

Hu距不变性

```
Hu不变距，Hu不变矩在图像旋转、缩放、平移等操作后，仍能保持矩的不变性，所以有时候用Hu不变距更能识别图像的特征
```

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404100959421.png" alt="image-20220404100959421" style="zoom:50%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404101020443.png" alt="image-20220404101020443" style="zoom:50%;" />

计算方法

moments()来计算图像中的中心矩(最高到三阶)，HuMoments()用于由中心矩计算Hu矩.



```
def match_shape_demo():
    src1 = cv.imread("./picture/m2.jpg",cv.IMREAD_GRAYSCALE)
    src2 = cv.imread("./picture/m3.jpg",cv.IMREAD_GRAYSCALE)
    cv.imshow("input-1",src1)
    cv.imshow("input-2",src2)

    #二值化
    ret1,binary1=cv.threshold(src1,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU) #利用OSTU,自动二值化
    ret2, binary2 = cv.threshold(src2, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 利用OSTU,自动二值化
    cv.imshow("binary1", binary1)
    cv.imshow("binary2", binary2)
    contours1,hierachy2=cv.findContours(binary1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours2,hierachy2=cv.findContours(binary2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    mm1=cv.moments(contours1[0])
    mm2=cv.moments(contours2[0])

    hu1=cv.HuMoments(mm1)
    hu2=cv.HuMoments(mm2)
    distance=cv.matchShapes(hu1,hu2,cv.CONTOURS_MATCH_I1,0)  #数值越小，差距越小
    print(distance)
```





## 第十四节：直线检测

霍夫直线检测

API知识点



```
#第十四节：直线检测
def hough_line_demo():
    src =cv.imread("./picture/route.jpg")
    cv.imshow("input",src)
    src=cv.GaussianBlur(src,(3,3),0)

    #边缘提取 #阈值
    edges=cv.Canny(src,180,300,apertureSize=3)
    cv.imshow("edges",edges)

    lines=cv.HoughLines(edges,1,np.pi/180,150,None,0,0)
    if lines is not None:
        for i in range(0,len(lines)):
            rho=lines[i][0][0]
            theta=lines[i][0][1]
            a=math.cos(theta)
            b=math.sin(theta)
            x0=a*rho
            y0=b*rho
            pt1=(int(x0+500*(-b)),int(y0+a*500))
            pt2=(int(x0-500*(-b)),int(y0-a*500))
            cv.line(src,pt1,pt2,(0,0,255),2,8,0)
    cv.imshow("hough-line",src)

    image=np.copy(src)
    linesP=cv.HoughLinesP(edges,1,np.pi/180,100,None,50,10)
    if lines is not None:
        for i in range(0, len(linesP)):
            l=linesP[i][0]
            cv.line(src,(l[0],l[1]),(l[2],l[3]),(255,0,0),2,8,0)
    cv.imshow("hough-line_s",image)
```









## 第二十节：视频读写

视频文件读写

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404143051694.png" alt="image-20220404143051694" style="zoom:50%;" />

摄像头读写

>摄像头读写属性
>
>Videocapture函数，摄像头index从0开始
>
>

视频帧率与宽高

API知识点



capture=cv.VideoCapture("./picture/vtest.avi")#读视频文件

 capture=cv.VideoCapture(0)   #读摄像头

#写出保存视频    out=cv.VideoWriter("C:/Users/LENOVO/Desktop/mv/test.mp4",cv.VideoWriter_fourcc('D','I','V','x'),15,(np.int(width),np.int(height)),True)

```
#第二十节：视频读写
def video_io_demo():
    capture=cv.VideoCapture("./picture/vtest.avi")#读视频文件
    height=capture.get(cv.CAP_PROP_FRAME_HEIGHT)  #获取视频高度
    width=capture.get(cv.CAP_PROP_FRAME_WIDTH)  #获取视频宽度
    count = capture.get(cv.CAP_PROP_FRAME_COUNT)  #获取帧数
    fps=capture.get(cv.CAP_PROP_FPS)     #获得帧率 ：每秒钟的帧数
    print(height,width,count,fps)
    while(True):
        ret,frame=capture.read()
        if ret is True:
            cv.imshow("video-input",frame)
            c=cv.waitKey(100)
            if c==27:
                break #ESC

#第二十节：摄像头读写
def camera_i_demo():
    capture=cv.VideoCapture(0)   #读摄像头
    height=capture.get(cv.CAP_PROP_FRAME_HEIGHT)  #获取视频高度
    width=capture.get(cv.CAP_PROP_FRAME_WIDTH)  #获取视频宽度
    count = capture.get(cv.CAP_PROP_FRAME_COUNT)  #获取帧数
    fps=capture.get(cv.CAP_PROP_FPS)     #获得帧率 ：每秒钟的帧数
    print(height,width,count,fps)

    #写出保存视频
    out=cv.VideoWriter("C:/Users/LENOVO/Desktop/mv/test.mp4",cv.VideoWriter_fourcc('D','I','V','x'),15,(np.int(width),np.int(height)),True)
    while(True):
        ret,frame=capture.read()
        if ret is True:
            cv.imshow("video-input",frame)
            out.write(frame)  #写出保存视频
            c=cv.waitKey(100)
            if c==27:
                break #ESC
```







## 第二十一节：视频帧处理

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404143111697.png" alt="image-20220404143111697" style="zoom: 67%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404143143409.png" alt="image-20220404143143409" style="zoom:50%;" />





```
#第二十一节：视频帧处理
def process_frame(frame,type):
    if type==0:
        gray =cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
        binary=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10)  #自适应阈值
        return binary
    if type ==1:
        dst =cv.GaussianBlur(frame,(0,0),25)
        return  dst
    if type==2:  #unsharp_mask
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        im=cv.filter2D(frame,-1,kernel)
        aw=cv.addWeighted(im,2,cv.GaussianBlur((frame,(0,0),15),-2,128))
        return aw
    else:
        return frame


def camera_i_demo1():
    capture=cv.VideoCapture(0)   #读摄像头
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)  # 获取视频高度
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)  # 获取视频宽度
    #写出保存视频
    out=cv.VideoWriter("C:/Users/LENOVO/Desktop/mv/test.mp4",cv.VideoWriter_fourcc('D','I','V','x'),15,(np.int(width),np.int(height)),True)
    while(True):
        ret,frame=capture.read()
        if ret is True:
            cv.imshow("video-input",frame)
            type=0
            result=process_frame(frame,type)
            cv.imshow("video-result",result)
            c=cv.waitKey(100)
            if c==27:
                break #ESC
            out.write(frame)  # 写出保存视频
```





## 第二十二节：实时人脸检测

HAAR 与LBP人脸检测文件

>HAAR  
>
>C:\Users\LENOVO\Desktop\opencv学习\opencv-4.x\data\haarcascades

人脸检测原理

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404154644989.png" alt="image-20220404154644989" style="zoom:50%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404154524163.png" alt="image-20220404154524163" style="zoom:50%;" />

API参数

```
#设置级联器：
#face_detecor=cv.CascadeClassifier("./picture/haarcascade_frontalface_alt_tree.xml")
face_detecor =cv.CascadeClassifier("picture/haarcascade_frontalface_default.xml")
def detect_face(frame):
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gray=cv.equalizeHist(gray)  #利用直方图来增强对比度
    faces=face_detecor.detectMultiScale(gray,1.2,4,minSize=(100,100),maxSize=(400,400))  #检测的最小，最大值
    for x,y,w,h in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2,8,0)
    return frame

def camera_i_demo2():
    capture=cv.VideoCapture(0)   #读摄像头
    height = 600
    width =480
    #写出保存视频
    out=cv.VideoWriter("C:/Users/LENOVO/Desktop/mv/test.mp4",cv.VideoWriter_fourcc('D','I','V','x'),15,(np.int(width),np.int(height)),True)
    while(True):
        ret,frame=capture.read()
        if ret is True:
            result=detect_face(frame)
            cv.imshow("video-result",result)
            c=cv.waitKey(100)
            if c==27:
                break #ESC
            out.write(frame)  # 写出保存视频
```





## 第二十三节：背景分析

背景分析基本原理

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404173824654.png" alt="image-20220404173824654" style="zoom:50%;" />

常用方法：

KNN

GMM

BackgroundSubtractorMOG2用到的是基于自适应混合高斯背景建模的背景减除法，相对于BackgroundSubtractorMOG，其具有更好的抗干扰能力，特别是光照变化。

>***\*ps：\****BackgroundSubtractorMOG2等一些背景减除法、帧差法仅仅做运动检测，网上经常有人做个运动检测，再找个轮廓，拟合个椭圆就说跟踪了，混淆了概念，凡是没有建立帧与帧之间目标联系的，没有判断目标产生和目标消失的都不能算是跟踪吧。

![image-20220404163648516](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404163648516.png)





>
>
>getStructuringElement函数会返回指定形状和尺寸的结构元素。
>
>Mat getStructuringElement(int shape, Size esize, Point anchor = Point(-1, -1));
>这个函数的第一个参数表示内核的形状，有三种形状可以选择。
>
>矩形：MORPH_RECT;
>
>交叉形：MORPH_CROSS;
>
>椭圆形：MORPH_ELLIPSE;
>
>第二和第三个参数分别是内核的尺寸以及锚点的位置。一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得getStructuringElement函数的返回值。





![image-20220404164450911](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404164450911.png)





[(7条消息) 视频背景分析_可欣の扣得儿的博客-CSDN博客](https://blog.csdn.net/aspirinLi/article/details/106772900)



```
##第二十三节：背景分析
def background_demo():
    capture = cv.VideoCapture(0)  # 读摄像头
    bgfg=cv.createBackgroundSubtractorMOG2()  #建立模型
    k=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    while (True):
        ret, frame = capture.read()
        if ret is True:
            cv.imshow("video-input",frame)
            mask=bgfg.apply(frame)  #把模型应用到frame上
            bg_image = bgfg.getBackgroundImage()  #获得背景
            mask=cv.morphologyEx(mask,cv.MORPH_OPEN,k)  #获得前景mask
            cv.imshow("video-result",mask)
            cv.imshow("backgroung",bg_image)
            c = cv.waitKey(100)
            if c == 27:
                break  # ESC
```





## 第二十四：颜色对象跟踪

色彩空间变换与mask提取

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404173852394.png" alt="image-20220404173852394" style="zoom:50%;" />

形态学处理

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404173903656.png" alt="image-20220404173903656" style="zoom:50%;" />

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404173918720.png" alt="image-20220404173918720"  />

```
#第二十四：颜色对象跟踪
def color_object_trace():
    capture = cv.VideoCapture(0)  # 读摄像头

    k=cv.getStructuringElement(cv.MORPH_RECT,(5,5))  #用来给mask去噪
    while (True):
        ret, frame = capture.read()
        if ret is True:
            #利用颜色图像获得mask
            cv.imshow("video-input", frame)
            hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
            mask=cv.inRange(hsv,(0,0,0),(180,255,46)) #获取指定颜色:黑色 范围内的mask
            mask=cv.morphologyEx(mask,cv.MORPH_OPEN,k)
            cv.imshow("mask",mask)
            #利用mask,标记原图
            contours, hierachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓查找
            print("total number", len(contours))  #获取轮廓

            #获取最大轮廓
            max=0
            temp=0
            index=-1
            for i in range(len(contours)):
                x,y,w,h=cv.boundingRect(contours[i])
                temp=w*h
                if temp>max:
                    max=temp
                    index=i
            if index>=0:#防止一开始为空
                x, y, w, h = cv.boundingRect(contours[index])
                cv.rectangle(frame,(x,y),(x+w,h+y),(0,0,255),2,4,0)
            cv.imshow("trace-object-demo",frame)
            c = cv.waitKey(100)
            if c == 27:
                break  # ESC
```





## 第二十五节：几何识别

多边形逼近几何形状识别

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404194729420.png" alt="image-20220404194729420" style="zoom:67%;" />

基本流程

<img src="C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20220404194717631.png" alt="image-20220404194717631" style="zoom:67%;" />

