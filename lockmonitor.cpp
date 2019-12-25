
#include "lockmonitor.h"
#include <cmath>
//#include "http_client.h"
#include <stdio.h>
#include <iostream>
//#include "MatToQImage.h"
//#include "HCNetSDK.h"
//#include "plaympeg4.h"
#include <time.h>
//#include<opencv2\core\core.hpp>
//#include<opencv2\opencv.hpp>   
//#include"global.h"
//#include <opencv2\imgproc\types_c.h>
//#include"caffe_classfier.h"

#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"


#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"


using namespace std ;
using namespace tensorflow;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace cv;
int time_flash = 1000;// 刷新间隔
//将mat转换为tensor
/*
Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    vector<Tensor> outTensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return Status::OK();
}*/
Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);


    return Status::OK();
}
Status loadGraph(const string &graph_file_name,
                 unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}


LockMonitor::LockMonitor(QWidget *parent)//构造函数
    : QMainWindow(parent)
{
    //dia_progress.close();
    //    dia_progress  = new QProgressDialog();
    //    dia_progress->setWindowTitle("正在识别");//设置窗口标题
    //    dia_progress->setLabelText("Running...");//提示标签
    //    dia_progress->setMinimum(0);//设置最小值
    //    dia_progress->setMaximum(0);//设置最大值

    ui.setupUi(this);
    ui. SMS_status_label->hide();
    ui. label->hide();
    ui. label_2->hide();
    ui. label_connected_status->hide();

    ui. SMS_button->hide();
    ui. pushButton_3->hide();
    ui. pushButton_stop_preview->hide();
    ui. SMS_button->hide();



    //extern Classifier*  classifier;//初始化caffe模型
    //classifier = new Classifier();
    //imageBufferSize = 2;//定义缓冲区的大小
    connect(ui.capture_img_button, SIGNAL(clicked()), this, SLOT(capture_one_img()));
    //connect(ui.pushButton_login, SIGNAL(clicked()), this, SLOT(login_camera_system()));
    //connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(open_camera()));
    //connect(ui.pushButton_2, SIGNAL(clicked()), this, SLOT(stopProgram()));
    connect(ui.SMS_button, SIGNAL(clicked()), this, SLOT(SMS_button_slot()));
    connect(ui.pushButton_stop_preview, SIGNAL(clicked()), this, SLOT(slot_stop_preview()));//停止预览
    connect(ui.pushButton_3, SIGNAL(clicked()), this, SLOT(set_camera_info()));
    connect(ui.pushButton_strat_preview, SIGNAL(clicked()), this, SLOT(strat_Preview()));
    // connect(&capimg_thread, SIGNAL(send_lock_info(int, float, float)), this, SLOT(send_SMS_slot(int, float, float)));
    connect(this, SIGNAL(send_lock_info(int, float, float)), this, SLOT(slot_update_lock_status(int, float, float)));

    QPixmap *img = new QPixmap("./Resources/11.png");
    img->scaled(ui.label_connected_status->size(), Qt::IgnoreAspectRatio);
    // ui.label_connected_status->setPixmap(*img);
    delete img;
    ui.SMS_status_label->setPixmap(QPixmap("./Resources/sms_close.png"));
    // ui.label_connected_status->show();
    ui.frameLabel->setStyleSheet("background-color:rgb(204,204,204)");
    ui.frameLabel_2->setStyleSheet("background-color:rgb(204,204,204)");
    //  dialog = new Open_camera_dialog();
    //connect(dialog->ui.okButton, SIGNAL(clicked()), this, SLOT(login_camera_system()));
    // dialog->setModal(true);
    //dialog->show();
    //初始化两个监控
    // vcp_3101=cv::VideoCapture("rtsp://admin:admin12345@222.195.151.251/Streaming/Channels/3001");
    // vcp_3171=cv::VideoCapture("rtsp://admin:admin12345@222.195.151.251/Streaming/Channels/2901");
    // vcp_3171.set(CV_CAP_PROP_BUFFERSIZE,1);
    // vcp_3101.set(CV_CAP_PROP_BUFFERSIZE,1) ;
    //double rate= vcp_310.get(CV_CAP_PROP_FPS);
    //qDebug() << "rate:" << rate;

    ////////////////////////////////////////////////////////////////////////////////////
    customPlot_317 = ui.tab_317;//获得画布
    customPlot_310 = ui.tab_310;//获得画布
    //初始化开关门数据  可以从文件里边进行读取
    lock_data_310 = QVector<QVector<int>>(7,QVector<int>(96,0));
    lock_data_317 = QVector<QVector<int>>(7,QVector<int>(96,0));
    this->readDataFromFile(lock_data_310,lock_data_317);
    manage = new QNetworkAccessManager(this);//初始化发出http请求的对象
    //初始化画布
    this->huabu_chushihua(customPlot_317);
    this->huabu_chushihua(customPlot_310);
    //设置定时器
    qtimer = new QTimer(this);
    connect(qtimer, SIGNAL(timeout()), this, SLOT(lockDataShow_run()));
    qtimer->start(1*60*1000);
    lRealPlayHandle = 1;
    //customPlot_317->xAxis->setTickLabelType(QCPAxis::ltDateTime);


    //    dia_progress->hide();
    ////////////////////////////////////////////////////////////////////////////////////
    //    dia_progress->hide();
}
//画布初始化
void LockMonitor::huabu_chushihua(QCustomPlot* customPlot){


    customPlot->axisRect()->setRangeZoom(Qt::Horizontal);
    QLinearGradient gradient(0, 0, 0, 400);
    gradient.setColorAt(0, QColor(90, 90, 90));
    gradient.setColorAt(0.38, QColor(105, 105, 105));
    gradient.setColorAt(1, QColor(70, 70, 70));
    customPlot->setBackground(QBrush(gradient));

    QVector<double> ticks;
    QVector<QString> labels;
    ticks << 1 << 2 << 3 << 4 << 5 << 6 << 7;
    labels << "Mon" << "Tue" << "Wed" << "Thu" << "Fri" << "Sat" << "Sun";
    QSharedPointer<QCPAxisTickerText> textTicker(new QCPAxisTickerText);
    textTicker->addTicks(ticks, labels);
    customPlot->yAxis->setTicker(textTicker);
    //customPlot->yAxis->setTickLabelRotation(60);
    customPlot->yAxis->setSubTicks(false);
    customPlot->yAxis->setTickLength(0, 4);
    customPlot->yAxis->setRange(0, 8);
    customPlot->yAxis->setBasePen(QPen(Qt::white));
    customPlot->yAxis->setTickPen(QPen(Qt::white));
    customPlot->yAxis->grid()->setVisible(true);
    customPlot->yAxis->grid()->setPen(QPen(QColor(130, 130, 130), 0, Qt::DotLine));
    customPlot->yAxis->setTickLabelColor(Qt::white);
    customPlot->yAxis->setLabelColor(Qt::white);

    // prepare y axis:
    //设置x轴的时间刻度
    QSharedPointer<QCPAxisTickerTime> timeTicker(new QCPAxisTickerTime);
    customPlot->xAxis->setTicker(timeTicker);
    customPlot->xAxis->setRange(0, 60*24.5);
    timeTicker->setTimeFormat("%m:%s");
    customPlot->xAxis->ticker()->setTickCount(10);//11个主刻度
    customPlot->xAxis->ticker()->setTickStepStrategy(QCPAxisTicker::tssReadability);//可读性优于设置
    //customPlot->xAxis->setRange(0, 98);
    customPlot->xAxis->setPadding(12); // a bit more space to the left border
    //customPlot->xAxis->setLabel("Power Consumption in\nKilowatts per Capita (2007)");
    customPlot->xAxis->setBasePen(QPen(Qt::white));
    customPlot->xAxis->setTickPen(QPen(Qt::white));
    customPlot->xAxis->setSubTickPen(QPen(Qt::white));
    customPlot->xAxis->grid()->setSubGridVisible(true);
    customPlot->xAxis->setTickLabelColor(Qt::white);
    customPlot->xAxis->setLabelColor(Qt::white);
    customPlot->xAxis->grid()->setPen(QPen(QColor(130, 130, 130), 0, Qt::SolidLine));
    customPlot->xAxis->grid()->setSubGridPen(QPen(QColor(130, 130, 130), 0, Qt::DotLine));
    // setup legend:

    customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom|QCP::iSelectPlottables);

}
//画图
void LockMonitor::lockDataShow(const QVector<QVector<int>>& vec,QCustomPlot* customPlot)
{
    //清空之前所有的柱
    customPlot->clearPlottables( );
    QVector<QCPBars *> bars;
    QCPBars * last = nullptr;
    for(int i = 0;i<vec.size();++i){
        QVector<double> date(7,1.0);
        date[i] = i+1;
        for(int j = 0;j<vec[0].size();++j){
            QVector<double> houdu(7,0.0);
            houdu[i] = 15.0;
            int num = vec[i][j];
            QCPBars * bar =new QCPBars(customPlot->yAxis, customPlot->xAxis);

            bar->setAntialiased(false);
            bar->setStackingGap(0.0);
            if(num ==0){
                bar->setPen(QPen(QColor(199,199,199).lighter(170)));
                bar->setBrush(QColor(199,199,199));
            }
            else if(num==1){
                bar->setPen(QPen(QColor(65,205,82).lighter(150)));
                bar->setBrush(QColor(65,205,82));
            }
            else{
                bar->setPen(QPen(QColor(242, 84,91).lighter(130)));
                bar->setBrush(QColor(242, 84,91));
            }
            while(j+1<vec[0].size()&&vec[i][j+1]==vec[i][j]){
                j++;
                houdu[i] = houdu[i]+15;
            }
            if(!bars.empty())
                bar->moveAbove(bars.back());
            bars.append(bar);
            bar->setData(date, houdu);

        }
        //qDebug()<<"666";
    }
    customPlot->replot();
    //customPlot->yAxis->setRange(0, 8.0+0.1*((++lRealPlayHandle)%2));
    //qDebug()<<bars.size();
}

LockMonitor::~LockMonitor()
{
    this->writeDataToFile(lock_data_310,lock_data_317);
    // vcp_3101.release();
    //  vcp_3171.release();
    //  delete dialog;
    //NET_DVR_Logout(lUserID);
    // NET_DVR_Cleanup();

}
//extern long Channel;
//extern long lUserID;
//extern short Port_317;//310和317d的通道号
//extern short Port_310;
//extern char *login_password;
//extern char *login_username;
//extern char *login_ip;
//extern unsigned short  login_port;
void LockMonitor::login_camera_system() {

}

void LockMonitor::strat_Preview() {  //测试预览槽函数
    //timer = new QTimer(this);
    // connect(timer,SIGNAL(timeout()),this,SLOT(rtsp_Preview()));
    // timer->start(time_flash);        //为避免出现延时累积，每5毫秒刷新界面
    rtsp_Preview();
}
void LockMonitor::rtsp_Preview() {//测试预览实现
    cv::VideoCapture vcp_310=cv::VideoCapture("rtsp://admin:admin12345@222.195.151.251/Streaming/Channels/3001");
    cv::VideoCapture vcp_317=cv::VideoCapture("rtsp://admin:admin12345@222.195.151.251/Streaming/Channels/2901");
    vcp_310>>src_frame_310; //从视频取帧
    // if(!src_frame_310.data)timer->stop();//如果取不到数据，终止计时器


    cv::Size dsize = cv::Size(ui.frameLabel_2->width(),
                              ui.frameLabel_2->height());
    cv::resize(src_frame_310, src_frame_310, dsize );

    if(src_frame_310.channels() == 3) {   // RGB image
        cvtColor(src_frame_310,src_frame_310,CV_BGR2RGB);
        qimg_310 = QImage((const uchar*)(src_frame_310.data),  //(const unsigned char*)
                          src_frame_310.cols,src_frame_310.rows,
                          src_frame_310.cols*src_frame_310.channels(),   //new add
                          QImage::Format_RGB888);
    }else {                     // gray image
        qimg_310 = QImage((const uchar*)(src_frame_310.data),
                          src_frame_310.cols,src_frame_310.rows,
                          src_frame_310.cols*src_frame_310.channels(),    //new add
                          QImage::Format_Indexed8);
    }
    //ui->label->clear();
    ui.frameLabel_2->setPixmap(QPixmap::fromImage(qimg_310));
    ////////////////////////////////////////////////////////////////////////////////////
    vcp_317>>src_frame_317; //从视频取帧
    if(!src_frame_317.data)timer->stop();//如果取不到数据，终止计时器


    dsize = cv::Size(ui.frameLabel->width(),
                     ui.frameLabel->height());
    cv::resize(src_frame_317, src_frame_317, dsize );

    if(src_frame_317.channels() == 3) {   // RGB image
        cvtColor(src_frame_317,src_frame_317,CV_BGR2RGB);
        qimg_317 = QImage((const uchar*)(src_frame_317.data),  //(const unsigned char*)
                          src_frame_317.cols,src_frame_317.rows,
                          src_frame_317.cols*src_frame_317.channels(),   //new add
                          QImage::Format_RGB888);
    }else {                     // gray image
        qimg_317 = QImage((const uchar*)(src_frame_317.data),
                          src_frame_317.cols,src_frame_317.rows,
                          src_frame_317.cols*src_frame_317.channels(),    //new add
                          QImage::Format_Indexed8);
    }
    //ui->label->clear();
    ui.frameLabel->setPixmap(QPixmap::fromImage(qimg_317));
    vcp_310.release();
    vcp_317.release();
}
void LockMonitor::set_camera_info() {//配置摄像头

}
void LockMonitor::stopProgram()
{

    cout << "device id is :"  << endl;

}
bool LockMonitor::SMS_is_open = false;
void  LockMonitor::SMS_button_slot() {//短息通知按钮点击事件

}
void  LockMonitor::SMS_try_to_send_slot() {//判断是否到时间了，检测一下试下？

}

void LockMonitor::capture_one_img()              //识别
{

    cv::VideoCapture vcp_310=cv::VideoCapture("rtsp://admin:admin12345@222.195.151.251/Streaming/Channels/3001");
    cv::VideoCapture vcp_317=cv::VideoCapture("rtsp://admin:admin12345@222.195.151.251/Streaming/Channels/2901");
    //    dia_progress->show();

    //dia_progress.exec();

    Mat frame;
    Mat frame2;
    vcp_310>>src_frame_310; //从视频取帧
    vcp_317>>src_frame_317; //从视频取帧
    if(!src_frame_317.data){
        qDebug()<<"get frame filaed...";
        return;
    }
    if(!src_frame_310.data){
        qDebug()<<"get frame filaed...";
        return;
    }

    frame  = src_frame_310(Rect(838,532,100,100));
    frame2  = src_frame_317(Rect(973,453,100,100));
    //qDebug()<<"get frame finish...";
    //   cv::imshow("image",frame);
    // cv::imshow("image2",frame2);
    string ROOTDIR = "/home/nvidia/Desktop/LockMonitor/";

    string GRAPH = "zxf.pb";

    // Set input & output nodes name
    // Load and initialize the model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        qDebug()<<"loadGraph(): ERROR";
        cout << "loadGraph(): ERROR" << loadGraphStatus;
        return;
    } else{
        qDebug()<<"model load success..";
        //strat_Preview();
    }
    tensorflow::TensorShape shape = tensorflow::TensorShape();

    // frame = cv::imread("/home/nvidia/Desktop/LockMonitor/628.jpg");
    //frame2 = cv::imread("/home/nvidia/Desktop/LockMonitor/1024.jpg");

    shape.AddDim(1);
    shape.AddDim(100);
    shape.AddDim(100);
    shape.AddDim(3);
    Tensor tensor;
    tensor = Tensor(tensorflow::DT_FLOAT, shape);
    Tensor tensor2 = Tensor(tensorflow::DT_FLOAT, shape);
    string inputLayer = "Placeholder";
    vector<string> outputLayer ={ "softmax_linear/softmax_linear_1"};

    std::vector<Tensor> outputs;
    std::vector<Tensor> outputs2;
    cvtColor(frame,frame,CV_BGR2RGB);
	//将mat转换为tensor,两张图两个tensor
    Status readTensorStatus = readTensorFromMat(frame, tensor);
    Status readTensorStatus2 = readTensorFromMat(frame2, tensor2);
    if (!readTensorStatus.ok()||!readTensorStatus2.ok()) {
        qDebug()<< "Mat->Tensor conversion failed: " ;
        return ;
    }
    LOG(INFO) <<tensor.shape()<<"111111111111111111111111111";
    outputs.clear();
    outputs2.clear();
    Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
    Status runStatus2 = session->Run({{inputLayer, tensor2}}, outputLayer, {}, &outputs2);
    if (!runStatus.ok()&&(!runStatus2.ok())) {
        qDebug()<<  "Running model failed: ";
        std::cout<<2222222222<<runStatus.ToString();
        return ;
    }

    else{
        qDebug()<<  "Running model test success";
    }
    qDebug()<<outputs.size();
    std::cout<<outputs[0].DebugString()<<endl;
    //outputs里边只有一个tensor 转换为float型的tensor 2表示二维
    auto tmap_pro = outputs[0].tensor<float, 2>();
    auto tmap_pro2 = outputs2[0].tensor<float, 2>();
    vector<float> result_310;
    vector<float> result_317;
    for(int i = 0;i<4;++i){
        qDebug()<<float(tmap_pro(0,i));
        qDebug()<<float(tmap_pro2(0,i));
        result_310.push_back(float(tmap_pro(0,i)));
        result_317.push_back(float(tmap_pro2(0,i)));
    }
    qtime = QTime::currentTime();
    time_t time_seconds = time(0);
    struct tm* now_time = localtime(&time_seconds);
    //星期几
    int weekday = (now_time->tm_wday+6)%7;
    qDebug()<<"week"<<weekday;//0代表星期一 以此类推
    //96中的第几个
    int index = qtime.hour()*4+(int)((qtime.minute()/15));
    qDebug()<<"index"<<index;
    if(result_310[0]>result_310[1])//310关
    {

        lock_data_310[weekday][index] = 2;
        emit send_lock_info(1, (float)exp(result_310[0])/(exp(result_310[0])+exp(result_310[1])), (float)exp(result_310[1])/(exp(result_310[0])+exp(result_310[1])));
    }
    //310开
    else{
        lock_data_310[weekday][index] = 1;
        emit send_lock_info(2, (float)exp(result_310[0])/(exp(result_310[0])+exp(result_310[1])), (float)exp(result_310[1])/(exp(result_310[0])+exp(result_310[1])));

    }

    if(result_317[0]>result_317[1])//317关
    {
        lock_data_317[weekday][index] = 2;
        emit send_lock_info(3, (float)exp(result_317[0])/(exp(result_317[0])+exp(result_317[1])), (float)exp(result_317[1])/(exp(result_317[0])+exp(result_317[1])));
    }
    //317开
    else{
        lock_data_317[weekday][index] = 1;
        emit send_lock_info(4, (float)exp(result_317[0])/(exp(result_317[0])+exp(result_317[1])), (float)exp(result_317[1])/(exp(result_317[0])+exp(result_317[1])));

    }
    for(int i = weekday*96+index+1;i<96*7;++i){

        int ii = (int)(i/96);
        int jj = i%96;
        lock_data_317 [ii][jj] = 0;
        lock_data_310 [ii][jj] = 0;


    }
    //判断是否需要提醒
    //时间节点可以设置为全局变量,表示到时间并且开关们状态不符合要求就会发出提醒
    int moning_hour = 7;
    int moning_min = 59;
    int night_hour = 21;
    int night_min = 34;
    if(weekday<5&&(qtime.hour()==night_hour||qtime.hour()==moning_hour)){
        QString s = u8"";
        if(qtime.hour()==night_hour&&qtime.minute()==night_min){//需要关门

            if(result_317[0]<result_317[1]){//317 开
                s = s+u8"317需要关门";
            }
            if(result_310[0]<result_310[1]){//310开
                s = s+u8",310需要关门";
            }
            if(s.length()>2){
                sendNoticeToPhone(s);
            }
        }
        else if(qtime.hour()==moning_hour&&qtime.minute()==moning_min){//需要开门
            if(result_317[0]>result_317[1]){//317 关
                s = s+u8"317需要开门";
            }
            if(result_310[0]>result_310[1]){//310关
                s = s+u8",310需要关门";
            }
            if(s.length()>2){
                sendNoticeToPhone(s);
            }
        }
    }
    //画图
    this->lockDataShow(lock_data_317,customPlot_317);
    this->lockDataShow(lock_data_310,customPlot_310);
    // dia_progress->hide();
    vcp_310.release();
    vcp_317.release();
}
void LockMonitor::updateFrame(const QImage &frame)
{
    ui.frameLabel->setPixmap(QPixmap::fromImage(frame));

}
void LockMonitor::slot_stop_preview()//停止预览
{

    timer->stop();
    ui.frameLabel_2->clear();
    ui.frameLabel->clear();

}

void LockMonitor::slot_update_lock_status(int lock_status, float pre, float pre2)//更新开关门状态图标
{

    float p_close = 100 * (pre / (pre + pre2));
    float p_open = 100 * (pre2 / (pre + pre2));
    //cout << "p_open is :" << p_open << endl;
    QPixmap *img0 = new QPixmap("./Resources/loading.png");
    QPixmap *img1 = new QPixmap("./Resources/lock.png");
    QPixmap *img2 = new QPixmap("./Resources/unlock.png");
    switch (lock_status)
    {
    case 0: {

        img0->scaled(ui.label_status_right_png->size(), Qt::KeepAspectRatioByExpanding);
        ui.label_status_right_png->setPixmap(*img0);
        ui.label_status_right_png->show();
        ui.label_status_left_png->setPixmap(*img0);
        ui.label_status_left_png->show();
        break;
    }

    case 1: {
        QString s1 = u8"当前状态A310关门！";
        QString s2 = "\n";
        QString s3 = u8"开门概率：";
        QString s4 = QString::number(p_open, 'g', 4);
        //qDebug() << "s4 is :" << s4;
        QString s5 = "%\n";
        QString s6 = u8"关门概率：";
        QString s7 = QString::number(p_close, 'g', 4);
        QString s8 = "%\n";
        ui.label_310->setText( s3 + s4 + s5 + s6 + s7 + s8);
        img1->scaled(ui.label_status_right_png->size(), Qt::KeepAspectRatioByExpanding);
        ui.label_status_right_png->setPixmap(*img1);
        ui.label_status_right_png->show();
        break; }
    case 2: {
        QString s1 = u8"当前状态A310开门！";
        QString s2 = "\n";
        QString s3 = u8"开门概率：";
        QString s4 = QString::number(p_open, 'g', 4);
        //qDebug() << "s4 is :" << s4;
        QString s5 = "%\n";
        QString s6 = u8"关门概率：";
        QString s7 = QString::number(p_close, 'g', 4);
        QString s8 = "%\n";
        ui.label_310->setText(s3 + s4 + s5 + s6 + s7 + s8);
        img2->scaled(ui.label_status_right_png->size(), Qt::KeepAspectRatioByExpanding);
        ui.label_status_right_png->setPixmap(*img2);
        ui.label_status_right_png->show();
        break; }
    case 3: {
        QString s1 = u8"当前状态B317关门！";
        QString s2 = "\n";
        QString s3 = u8"开门概率：";
        QString s4 = QString::number(p_open, 'g', 4);
        //qDebug() << "s4 is :" << s4;
        QString s5 = "%\n";
        QString s6 = u8"关门概率：";
        QString s7 = QString::number(p_close, 'g', 4);
        QString s8 = "%\n";
        ui.label_317->setText(s3 + s4 + s5 + s6 + s7 + s8);
        img1->scaled(ui.label_status_left_png->size(), Qt::KeepAspectRatioByExpanding);
        ui.label_status_left_png->setPixmap(*img1);
        ui.label_status_left_png->show();
        break; }
    case 4: {
        QString s1 = u8"当前状态B317开门！";
        QString s2 = "\n";
        QString s3 = u8"开门概率：";
        QString s4 = QString::number(p_open, 'g', 4);
        //qDebug() << "s4 is :" << s4;
        QString s5 = "%\n";
        QString s6 = u8"关门概率：";
        QString s7 = QString::number(p_close, 'g', 4);
        QString s8 = "%\n";
        ui.label_317->setText(s3 + s4 + s5 + s6 + s7 + s8);
        img2->scaled(ui.label_status_left_png->size(), Qt::KeepAspectRatioByExpanding);
        ui.label_status_left_png->setPixmap(*img2);
        ui.label_status_left_png->show();
        break; }
    default:
        break;
    }
    delete img0, img1, img2;
}
