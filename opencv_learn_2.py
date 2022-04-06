
import cv2 as cv
import numpy as np


#第一节：图像模糊
def blur_demo():
    src=cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    dst=cv.blur(src,(10,10))  #图像，ksize(卷积核大小)， 卷积核越大，所取越多数的平均值，模糊程度越厉害
    dst1 = cv.blur(src, (30, 1))
    cv.imshow("blur image",dst)
    cv.imshow("blur1 image", dst1)


#第二节：均值与高斯模糊
def gaussian_blur_demo():
    src = cv.imread("./picture/girl.png")
    cv.imshow("input", src)
    dst=cv.GaussianBlur(src,(5,5),0)   #src,ksize,sigmaX   ksize和sigmaX只需要添一个即可，它俩会自动去根据公式计算转化
    cv.imshow("guassian_blur",dst)
    dst1=cv.GaussianBlur(src,(0,0),10)
    cv.imshow("gassian_blur1",dst1)
    #当前面ksize设置了，后面的sigmaX就不起作用了


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


#添加噪声：
def noise_and_denoise():
    src=cv.imread("./picture/lena.jpg")
    # cv.imshow("input",src)
    src1=np.copy(src)
    #添加噪声  :椒盐噪声
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

    #添加噪声，高斯噪声
    gniose=np.zeros(src1.shape,src.dtype)
    mean=(15,15,15)  #噪声的均值
    sigam=(30,30,30)  #噪声的方差
    cv.randn(gniose,mean,sigam) #对该图片产生高斯噪声
    cv.imshow("ganssian_noise",gniose)
    dst1=cv.add(src1,gniose)
    cv.imshow("ganssian_image",dst1)

    #去除椒盐滤波
    #均值滤波
    result=cv.blur(src,(5,5))
    cv.imshow("result1",result)
    #高斯滤波
    result2=cv.GaussianBlur(src,(5,5),0)
    cv.imshow("result2",result2)
    #中值滤波 ：效果最好
    result3=cv.medianBlur(src,5)
    cv.imshow("result3",result3)

    #去除高斯滤波
    #使用非局部均值去噪声，效果比上面三种好
    result4=cv.fastNlMeansDenoising(dst1,None,15,15,25)
    cv.imshow("result4",result4)
    result5 = cv.fastNlMeansDenoisingColored(dst1, None, 15, 15,10, 25)
    cv.imshow("result4", result4)

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




#边缘提取  :Canny边缘提取算法
def edge_demo():
    src = cv.imread("./picture/lena.jpg")
    cv.imshow("input",src)
    edge=cv.Canny(src,150,300)  #低阈值   高阈值
    dst=cv.bitwise_and(src,src,mask=edge)  #mask 仅显示白色部分，当作模板，这样与出来，就原来的边缘彩色线条
    cv.imshow("edge",dst)


#锐化:Laplacian锐化
def sharpen_image():
    #Laplacian 算子
    lap_5=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    lap_9=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    src=cv.imread("./picture/girl.png")
    cv.imshow("input",src)

    des=cv.filter2D(src,cv.CV_8U,lap_5)
    cv.imshow("output",des)
def do_nothing():
    pass
def unsahrp_mask():
    src = cv.imread("./picture/girl.png")
    cv.imshow("input", src)
    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])  #Laplacian锐化 lap_5
    im=cv.filter2D(src,-1,kernel)
    cv.imshow("sharping",im)

    cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("input_weight", "result", 0,50,do_nothing)  # tracebar_name  window_name  value取值    count最大值     onchange是callback的方法
    cv.createTrackbar("Gassician_sigma", "result", 0, 30, do_nothing)  # tracebar_name  window_name  value取值    count最大值     onchange是callback的方法
    cv.createTrackbar("gass_blur_weight", "result", 0, 50,do_nothing)  # tracebar_name  window_name  value取值    count最大值     onchange是callback的方法
    cv.createTrackbar("add_light", "result", 0, 255,do_nothing)  # tracebar_name  window_name  value取值    count最大值     onchange是callback的方法

    while True:
        input_weight=cv.getTrackbarPos("input_weight", "result")*0.1
        Gassician_sigma=cv.getTrackbarPos("Gassician_sigma", "result")*0.1
        gass_blur_weight=cv.getTrackbarPos("gass_blur_weight", "result")*0.1
        add_light=cv.getTrackbarPos("gass_blur_weight", "result")
        aw=cv.addWeighted(src,input_weight, cv.GaussianBlur(src,(0,0),3),gass_blur_weight,add_light)
        cv.imshow("result",aw)
        t=cv.waitKey(10)
        if t==27:
            break#Esc

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




#模板匹配
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


#第七节：模板匹配
def match_template_change_methods():
    src=cv.imread("./picture/graf3.png")
    tpl=cv.imread("./picture/tpl.jpg")
    cv.imshow("src",src)
    cv.imshow("tpl",tpl)
    cv.namedWindow("match-result", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("methods", "match-result", 0, 4,do_nothing)  # tracebar_name  window_name  value取值    count最大值     onchange是callback的方法
    #内部默认设置的
    # TM_CCOEFF = 4
    #
    # TM_CCOEFF_NORMED = 5
    #
    # TM_CCORR = 2
    #
    # TM_CCORR_NORMED = 3
    #
    # TM_SQDIFF = 0
    #
    # TM_SQDIFF_NORMED = 1

    while True:
        method=cv.getTrackbarPos("methods", "match-result")
        if method==0:
            result = cv.matchTemplate(src, tpl, cv.TM_SQDIFF)
            cv.imshow("result", result)
        if method==1:
            result = cv.matchTemplate(src, tpl, cv.TM_SQDIFF_NORMED)
            cv.imshow("result", result)
        if method==2:
            result = cv.matchTemplate(src, tpl, cv.TM_CCORR)
            cv.imshow("result", result)
        if method==3:
            result = cv.matchTemplate(src, tpl, cv.TM_CCORR_NORMED )
            cv.imshow("result", result)
        if method==4:
            result = cv.matchTemplate(src, tpl, cv.TM_CCOEFF)
            cv.imshow("result", result)
        minv, maxv, min_loc, max_loc = cv.minMaxLoc(result)  # 这个矩阵的最小值，最大值，并得到最大值，最小值的索引
        th, tw = tpl.shape[:2]
        clone=np.copy(src)
        if method==0 or method==1:  #值越大相关性越差，故用最小值点
            cv.rectangle(clone, min_loc, (min_loc[0] + tw, min_loc[1] + th), (0, 0, 255), 2, 8, 0)
        else: #值越大，相关性越强，用最大值点
            cv.rectangle(clone, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 0, 255), 2, 8, 0)
        cv.imshow("match-result", clone)
        c=cv.waitKey(10)
        if c==27:
            break#ESC

def template_llk():
    src=cv.imread("./picture/llk.png")
    tpl=cv.imread("./picture/llk_tlp.jpg")
    src=cv.resize(src,(480,600),cv.INTER_CUBIC)
    cv.imshow("input",src)
    cv.imshow("tpl",tpl)
    th,tw=tpl.shape[:2]
    result = cv.matchTemplate(src, tpl, cv.TM_CCOEFF_NORMED)
    cv.imshow("result",result)
    t=0.6
    loc=np.where(result>t)  #找出所有大于0.85的点
    for pt in zip(*loc[::-1]):
        cv.rectangle(src,pt,(pt[0]+tw,pt[1]+th),(0,0,255),2,0,0)
    cv.imshow("llk-demo",src)


#第八节：直方图比较法
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



#第九节：直方图反向投影
def back_projection_demo():
    target=cv.imread("./picture/llk_tlp.jpg")
    sample=cv.imread("./picture/llk.png")
    cv.imshow("target",target)
    cv.imshow("sample",sample)
    target_hsv=cv.cvtColor(target,cv.COLOR_BGR2HSV)
    sample_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)

    #计算直方图
    tar_hist=cv.calcHist([target_hsv],[0,1],None,[32,32],[0,180,0,256],None)
    tar_hist=cv.normalize(tar_hist,None,0,255,cv.NORM_MINMAX)#归一化
    cv.imshow("zft",tar_hist)
    img_backPrj = cv.calcBackProject([sample_hsv], [0, 1],tar_hist,[0,180,0,256],1)
    cv.imshow("backprojection",img_backPrj)

    #用滤波器去掉噪点
    k=np.zeros((5,5),dtype=np.uint8)
    dst=cv.erode(img_backPrj,k)
    cv.imshow("backP",dst)



#第十节图像金字塔
#高斯金字塔
def pyramid_demo(image):
    # cv.imshow("input",image)
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

if __name__=="__main__":
    multiple_sample_template_match()
    cv.waitKey(0)
    cv.destroyAllWindows()