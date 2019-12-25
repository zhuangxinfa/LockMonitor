#include "lockmonitor.h"
#include <iostream>
#include <fstream> 
void  LockMonitor::writeDataToFile(QVector<QVector<int>> data,QVector<QVector<int>> data2){
    ofstream fout("data");
    for(int i = 0;i<data.size();++i){
        for(int j = 0;j<data[0].size();++j){
            fout << data[i][j] << " ";
        }
    }
    for(int i = 0;i<data2.size();++i){
        for(int j = 0;j<data2[0].size();++j){
            fout << data2[i][j] << " ";
        }
    }
    fout.close();
}
void LockMonitor::lockDataShow_run(){
    //定时器停止之后的槽函数
    qDebug()<<"time out run datection one time";
    //识别
    this->capture_one_img();
    //测试预览

    this->rtsp_Preview();
}
void LockMonitor::sendNoticeToPhone(QString s){
    QString http_addresss = u8"http://miaotixing.com/trigger?id=tPa9G4K";
    http_addresss.append(u8"&text=");
    http_addresss.append(s);
    QUrl url(http_addresss);//应该设置为全局变量,方便后序的修改
    manage->get(QNetworkRequest(QUrl(url)));
    qDebug()<<"sendNoticeToPhone"<<s;
}
void LockMonitor::readDataFromFile(QVector<QVector<int>>& data,QVector<QVector<int>> &data2){
    ifstream fin("data");
    for(int i = 0;i<data.size();++i){
        for(int j = 0;j<data[0].size();++j){
            fin>>data[i][j];
        }

    }
    for(int i = 0;i<data2.size();++i){
        for(int j = 0;j<data2[0].size();++j){
            fin>>data2[i][j];
        }}
    fin.close();
}
