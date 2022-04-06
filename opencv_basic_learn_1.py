#导入并显示一张图片
# encoding:utf-8
from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# #src=cv.imread("../3h/picture/lena.jpg")
# src=cv.imread("../3h/picture/lena.jpg",cv.IMREAD_GRAYSCALE)  #转为灰色图片
# print(src)
# # cv.namedWindow("input",cv.WINDOW_AUTOSIZE)  #自动适应其大小
# cv.imshow("input",src)  #在刚刚创建的nameWindow上显示该图片
# cv.waitKey(0)  #窗口等待时间
# # cv.destroyAllWindows() #将创建的所有窗口销毁


#第一课：读写
# endcoding="utf-8"
def  read_write_imge():
    src = cv.imread("./picture/lena.jpg",cv.IMREAD_GRAYSCALE) #加载灰度图像
    #BGR三个通道
    cv.imshow("input", src)
    cv.imwrite("./output_picture/lena.jpg",src)  #写出  #不支持gif



#第二节： 图像创建
def creation_image():
    src = cv.imread("./picture/lena.jpg", cv.IMREAD_GRAYSCALE)
    cv.imshow("input", src)
    dst=np.copy(src)  # 复制图像数组
    dst.fill(127)
    cv.imshow("dst",dst)


    #method2
    blank =np.zeros([400,400],dtype=np.uint8)
    blank.fill(255)
    cv.imshow("blank",blank)


    #method3
    t3= np.zeros([40000],dtype=np.uint8)
    t4 =np.reshape(t3,[200 ,200])
    cv.imshow("t4",t4)

    #method4  #创建与原图一样大小的。
    clone=np.zeros(src.shape,src.dtype)
    cv.imshow("clone",clone )
    cv.imwrite("./output_picture/clone.jpg", clone)

    #method5
    t5=np.random.random_sample([400,400])*255  #生成的随机数在0-1之间
    cv.imshow("t5",t5)  #因为只显示0-255整数
    t6=np.uint8(t5)
    cv.imshow("t6",t6)  #显示的是黑白的

    t7 = np.uint8(np.random.random_sample([400, 400,3]) * 255)  #变成三通道的
    cv.imshow("t7",t7)




#第三节：图形绘制
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

    src3=np.zeros([600,600,3],dtype=np.uint8)
    for i in range(10000):
        x1,y1,x2,y2=np.uint(np.random.rand(4)*600)  #np.random.rand() --> 生成指定维度的的[0,1)范围之间的随机数
                                #生成一维的四个数[ 53.54329724 470.81534894 341.15551469  67.13552128]
        b=np.random.randint(0,255)
        g=np.random.randint(0,255)
        r=np.random.randint(0,255)
        cv.line(src3,(x1,y1),(x2,y2),(b,g,r),4,cv.LINE_4,0)
        cv.imshow("src3",src3)
        c=cv.waitKey(20) #中间停顿20ms
        if c==27:
            break #Esc

#第四节：鼠标响应事件
def my_mouse_callback(event,x,y,flag,params):  #响应事件， 当前鼠标位置x,y， flag ,用户要传数据
    if event==cv.EVENT_FLAG_LBUTTON:   #如果双击鼠标
        b,g,r=np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)
        cv.circle(params,(x,y),150,(b,g,r),2,cv.LINE_4,0)
    if event==cv.EVENT_MBUTTONUP:
        cv.rectangle(params, (x, y), (x+100, y+100), (0, 0, 255), 2, cv.LINE_4, 0)

def mouse_demo():
    src=np.zeros((512,512,3),dtype=np.uint8)
    cv.namedWindow("mouse_demo",cv.WINDOW_AUTOSIZE)    #namedwondow 和setMouseCallback一定要是同一个窗口，不然会显示没有该窗口
    cv.setMouseCallback("mouse_demo",my_mouse_callback,src) #设置鼠标监听事件   #window_name  事件函数   param
    while True:
        cv.imshow("mouse_demo",src)
        c=cv.waitKey(20)
        if c==27:
            break  #ESC





#第五节：滑块


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
        print(src.shape)
#第六节：像素读写（pixel）
def pixel_demo():
    #BGR
    src =cv.imread("./picture/lena.jpg")
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
    cv.imwrite("./output_picture/Lena_BGR.png",src)

#第七节：单通道和多通道
def chanenels_demo():
    src=cv.imread("./picture/lena.jpg")
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
    cv.imwrite("./output_picture/Lena_BGR1.png",dst)

#第八节：镜像反转
def mirror_demo():
    #手动更改每个像素
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
    xs=cv.flip(src,0)  #沿着x轴翻转 # >0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
    ys=cv.flip(src,1)  #沿着y轴
    x_ys=cv.flip(src,-1) #x,y轴同时翻转
    cv.imshow("xs",xs)
    cv.imshow("ys",ys)
    cv.imshow("x_ys",x_ys)


#第九节 图像旋转
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



#第10节：图像的插值
#插值防缩大小
def resize_image():
    src=cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    h,w=src.shape[:2]
    dst=cv.resize(src,(h//2,w//2),interpolation=cv.INTER_NEAREST)  #image ,dsize(大小)，插值方法:线性插值
    dst1 = cv.resize(src, (h // 2, w // 2), interpolation=cv.INTER_CUBIC)  # image ,dsize(大小)，插值方法:双立法插值
    cv.imshow("dst",dst)
    cv.imshow("dst1",dst1)



#第十一节：图像平移
#平移
def translate_image():
    src=cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    w,h=src.shape[:2]
    M=np.float32([[1,0,100],[0,1,100]])   #平移矩阵
    dst=cv.warpAffine(src,M,(w,h))
    cv.imshow("result",dst)

#第十一节：图像算术运算
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


#第十一节：图像的逻辑运算：
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




#第十二节：图像亮度与对比度调整
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


if __name__=="__main__":
    lookup_table_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()