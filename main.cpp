#include "lockmonitor.h"
#include <QApplication>
#define _GLIBCXX_USE_CXX11_ABI 0
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    LockMonitor w;
    w.show();

    return a.exec();
}
