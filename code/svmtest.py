import sys
import time
from code import svm
import os
from sklearn.externals import joblib

def svmtest(model_path):
    path = sys.path[1]
    tbasePath = os.path.join(path, "data\\Mnist-image\\test\\")
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.clock()
    allErrCount = 0
    allErrorRate = 0.0
    allScore = 0.0
    #加载模型
    clf = joblib.load(model_path)
    for tcn in tcName:
        testPath = tbasePath + tcn
        # print("class " + tcn + " path is: {}.".format(testPath))
        tflist = svm.get_file_list(testPath)
        # tflist
        tdataMat, tdataLabel = svm.read_and_convert(tflist)
        print("test dataMat shape: {0}, test dataLabel len: {1} ".format(tdataMat.shape, len(tdataLabel)))

        # print("test dataLabel: {}".format(len(tdataLabel)))
        pre_st = time.clock()
        preResult = clf.predict(tdataMat)
        pre_et = time.clock()
        print("Recognition  " + tcn + " spent {:.4f}s.".format((pre_et - pre_st)))
        # print("predict result: {}".format(len(preResult)))
        errCount = len([x for x in preResult if x != tcn])
        print("errorCount: {}.".format(errCount))
        allErrCount += errCount
        score_st = time.clock()
        score = clf.score(tdataMat, tdataLabel)
        score_et = time.clock()
        print("computing score spent {:.6f}s.".format(score_et - score_st))
        allScore += score
        print("score: {:.6f}.".format(score))
        print("error rate is {:.6f}.".format((1 - score)))

    tet = time.clock()
    print("Testing All class total spent {:.6f}s.".format(tet - tst))
    print("All error Count is: {}.".format(allErrCount))
    avgAccuracy = allScore / 10.0
    print("Average accuracy is: {:.6f}.".format(avgAccuracy))
    print("Average error rate is: {:.6f}.".format(1 - avgAccuracy))


if __name__ == '__main__':
    path = sys.path[1]
    model_path=os.path.join(path,'model\\svm.model')
    svmtest(model_path)


