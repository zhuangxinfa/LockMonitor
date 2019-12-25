#ifndef LOCKMONITOR_H
#define LOCKMONITOR_H
//#include "Controller.h"
#include <QtWidgets/QMainWindow>
#include "ui_lockmonitor.h"
//#include"UI/open_camera_dialog.h"
#include <QString>
#include <QMessageBox>
#include <iostream>
//#include"capture_img.h"
#include <QTextCodec>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include "Controller.h"
#include <QtWidgets/QMainWindow>
#include "ui_lockmonitor.h"
//#include"UI/open_camera_dialog.h"
#include <QString>
#include <QMessageBox>
#include <iostream>
//#include"capture_img.h"
#include <QTextCodec>
#include <fstream>
#include <QDebug>
#include<QTimer>
#include<QProgressDialog>
#include"qcustomplot.h"
#include <QTime>
#include <ctime>
#include <QNetworkAccessManager>    //加载网络请求头文件
#include <QNetworkReply>
#include <QNetworkRequest>      //加载发送请求头文件
using namespace std ;

class LockMonitor : public QMainWindow
{
    Q_OBJECT

public:
    QProgressDialog* dia_progress;
    explicit LockMonitor(QWidget *parent = 0);
    ~LockMonitor();
	//定时画图
	QTimer *qtimer ;
	QCustomPlot* customPlot_317;//获得画布
	QCustomPlot* customPlot_310;//获得画布
	QVector<QVector<int>> lock_data_310;//开关门数据数组
	QVector<QVector<int>> lock_data_317;//开关门数据数组
	void writeDataToFile(QVector<QVector<int>> lock_data_310,QVector<QVector<int>> lock_data_317);
	void readDataFromFile(QVector<QVector<int>>& lock_data_310,QVector<QVector<int>>& lock_data_317);
	QTime qtime;   //时间
    //void LockMonitor::handle_func(std::string rsp);
    private slots:
	//画布初始化
	void lockDataShow_run();
	void huabu_chushihua(QCustomPlot* customPlot_310);
		void lockDataShow( const QVector<QVector<int>> &lock_data_310,QCustomPlot* customPlot_310);
        void login_camera_system();
        //void open_camera();
        void stopProgram();
        void set_camera_info();
        void strat_Preview();
        void rtsp_Preview();
        void capture_one_img();
        void slot_update_lock_status(int status_lock,float p1,float p2);
        void slot_stop_preview();

    private slots:
        void updateFrame(const QImage &frame);
        //void updateFrame(const cv::Mat &frame);
        void SMS_button_slot();
        void SMS_try_to_send_slot();
      //  void send_SMS_slot(int status_lock, float p1, float p2);

private:
		QNetworkAccessManager *manage;//给手机发出http请求的对象
		void sendNoticeToPhone(QString s);//给手机发出通知的函数
       // cv::VideoCapture vcp_3101;
       // cv::VideoCapture vcp_3171;
        cv::Mat src_frame_310;
        cv::Mat src_frame_317;
        QImage qimg_317;
        QImage qimg_310;
     QTimer *timer;//用于预览界面的刷新的//计时器
    static bool SMS_is_open;
  //  Open_camera_dialog *dialog;
    long lRealPlayHandle;
    long lRealPlayHandle2;
    Ui::LockMonitorClass ui;
    //Ui::open_camera_dialog open_camera_dialog;
//	Controller *controller;
    //int sourceWidth;
    //int sourceHeight;
    //int imageBufferSize;
    //bool isCameraConnected;
    //Capture_img capimg_thread;
signals:
    void send_lock_info(int,float,float);
};

#endif // LOCKMONITOR_H
