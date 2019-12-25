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
#include <QNetworkAccessManager>    //������������ͷ�ļ�
#include <QNetworkReply>
#include <QNetworkRequest>      //���ط�������ͷ�ļ�
using namespace std ;

class LockMonitor : public QMainWindow
{
    Q_OBJECT

public:
    QProgressDialog* dia_progress;
    explicit LockMonitor(QWidget *parent = 0);
    ~LockMonitor();
	//��ʱ��ͼ
	QTimer *qtimer ;
	QCustomPlot* customPlot_317;//��û���
	QCustomPlot* customPlot_310;//��û���
	QVector<QVector<int>> lock_data_310;//��������������
	QVector<QVector<int>> lock_data_317;//��������������
	void writeDataToFile(QVector<QVector<int>> lock_data_310,QVector<QVector<int>> lock_data_317);
	void readDataFromFile(QVector<QVector<int>>& lock_data_310,QVector<QVector<int>>& lock_data_317);
	QTime qtime;   //ʱ��
    //void LockMonitor::handle_func(std::string rsp);
    private slots:
	//������ʼ��
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
		QNetworkAccessManager *manage;//���ֻ�����http����Ķ���
		void sendNoticeToPhone(QString s);//���ֻ�����֪ͨ�ĺ���
       // cv::VideoCapture vcp_3101;
       // cv::VideoCapture vcp_3171;
        cv::Mat src_frame_310;
        cv::Mat src_frame_317;
        QImage qimg_317;
        QImage qimg_310;
     QTimer *timer;//����Ԥ�������ˢ�µ�//��ʱ��
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
