
import cv2 as cv
import numpy as np


def do_nothing(a):
    pass
#第一节：图像二值化
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


#第二节：图像二值化
def threshold_segmentation_demo():
    src=cv.imread("./picture/lena.jpg",cv.IMREAD_GRAYSCALE) #灰度图像
    cv.imshow("input",src)
    #二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY
    t=50
    ret,dst=cv.threshold(src,t,255,cv.THRESH_BINARY) #二值化分割    ret:阈值， dst:二值化后的图像
    ret1, dst1 = cv.threshold(src, t, 255, cv.THRESH_TRUNC)
    ret2, dst2= cv.threshold(src, t, 255, cv.THRESH_TOZERO_INV)
    cv.imshow("threshold_segmentation",dst)
    cv.imshow("threshold_segmentation1", dst1)
    cv.imshow("threshold_segmentation2", dst2)
    print(ret)



#第三节：全局阈值
def threshold_segmentation_demo1():
    src=cv.imread("./picture/lena.jpg",cv.IMREAD_GRAYSCALE) #灰度图像
    cv.imshow("input",src)
    #二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY

    ret1, dst1 = cv.threshold(src, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    ret2, dst2= cv.threshold(src, 0, 255, cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
    cv.imshow("threshold_segmentation1", dst1)
    cv.imshow("threshold_segmentation2", dst2)
    print(ret1)
    print(ret2)

def threshold_method_demo():
    src = cv.imread("./picture/text.jpg", cv.IMREAD_GRAYSCALE)  # 灰度图像
    cv.imshow("input", src)
    # 二值化函数   cv.threshold() src:必须为灰度图像， thresh:阈值，大于为255，小于为0 可手动填入，也可利用算法查找   maxvalue:变成二值图像最大值为多少，一般255   type:可以是多个选项的组合，一般cv.THRESH_BINARY
    dst1 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,10,None)
    dst2 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,25,10)
    cv.imshow("threshold_segmentation1", dst1)
    cv.imshow("threshold_segmentation2", dst2)



#第五节：去噪声对二值化的影响

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

#第六节：连通组件扫描
def connected_components_demo():
    src = cv.imread("./picture/rice_noise.jpg", cv.IMREAD_GRAYSCALE)  # 灰度图像
    cv.imshow("input", src)
    src = cv.medianBlur(src, 5)  #中值去噪
    res,dst1 = cv.threshold(src,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)  #OTSU阈值
    cv.imshow("threshold_segmentation2", dst1)


    output=cv.connectedComponents(dst1,ltype=cv.CV_32S)  #ltype输出的数据的类型
    # print(output)  #标记图(26, array([[0, 0, 0, ..., 0, 0, 0],...[0, 0, 0, ..., 0, 0, 0]], dtype=int32))
    num_lables=output[0]
    lables=output[1] #标记图  #每一个连通的区域数值不同

    #生成随机颜色
    colors=[(0,0,0)]#保证背景黑色
    for i in range(num_lables):
        b=np.random.randint(0,256)
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        colors.append((b,g,r))

    #根据标识图，为一个图片上色，颜色为该位置的标记号，对应的随机颜色
    h,w=dst1.shape
    image=np.zeros((h,w,3),dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row,col]=colors[lables[row,col]]
    cv.imshow("colored label",image)





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


#第七节：发现绘制轮廓
def find_contours_demo():
    src = cv.imread("./picture/coins.jpg")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src,(3,3),0)#高斯去噪
    gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    res,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)  #OTSU阈值
    cv.imshow("threshold_segmentation2", binary)


    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv.drawContours(src ,contours, i, (0, 0, 255), 2)
    cv.imshow("contpurs", src)


#第八节：图像测量：面积，周长
def measure_contours_demo():
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


#第十三节：轮廓匹配
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
    distance=cv.matchShapes(hu1,hu2,cv.CONTOURS_MATCH_I1,0)
    print(distance)


import math
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

#第二十五节：几何识别
def analysis():
    src = cv.imread("./picture/image_detect.jpg")
    cv.imshow("input", src)
    h,w,ch=src.shape

    result=np.zeros((h,w,ch),dtype=np.uint8)
    #二值化图像
    gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

    cv.imshow("binary image",binary)

    #获得轮廓
    contours,hierachy=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(contours)):
        #提取绘制轮廓
        cv.drawContours(result,contours,cnt,(0,255,0),2)

        #轮廓逼近
        epsilon=0.01*cv.arcLength(contours[cnt],True)
        approx=cv.approxPolyDP(contours[cnt],epsilon,True)

        #分析几何形状
        corner=len(approx)
        shape_type=""
        #求中心位置：
        mm = cv.moments(contours[cnt])
        m00 = mm["m00"]
        m10 = mm["m10"]
        m01 = mm["m01"]
        cx = np.int(m10 / m00)
        cy = np.int(m01 / m00)
        cv.circle(result,(cx,cy),3,(0,0,255),-1)
        if corner==3:
            cv.putText(result,"triangle",(cx-10,cy),cv.FONT_HERSHEY_PLAIN,1,(255,0,0),2,0,0)
        elif corner == 4:
            cv.putText(result, "rectangle", (cx - 10, cy), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, 0, 0)
        elif 4<corner <10:
            cv.putText(result, "polygons", (cx - 10, cy), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, 0, 0)
        else:
            cv.putText(result, "circles", (cx - 10, cy), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, 0, 0)

        #颜色分析:颜色就是该点的BRG
        color=src[cy,cx]
        color_str=str(color)
        x, y, w, h = cv.boundingRect(contours[cnt])  #获得位置
        cv.putText(result,color_str,(cx,y+h+10),cv.FONT_HERSHEY_PLAIN,1,(255,255,0),2,0,0)

        #计算面积与周长：
        p=cv.arcLength(contours[cnt],True)
        area=cv.contourArea(contours[cnt])
        s="area:%d,length:%d"%(p,area)
        cv.putText(result, s, (cx, y + h + 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, 0, 0)

    cv.imshow("Analysis Result",result)


if __name__=="__main__":
    background_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()