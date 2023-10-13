import os
import time
import dpkt
import socket
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from utils.dnsprocess import ipdnsdirect, IP_DNS_DIRECT


def walkFile(file):
    filepatlist = []
    for root, dirs, files in os.walk(file):
        # root 表示当前访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        for f in files:
            filepath = os.path.join(root, f)
            filepatlist.append(filepath)
    return filepatlist


# 读取pcapdroid抓取的pcap文件
def createfrompcap(pcapfile):
    srcip = ''
    dstip = ''
    srcport = 0
    dstport = 0
    protocol_index = 0
    flow_unique = set()
    flowinpcap = []

    f = open(pcapfile, mode='rb')
    print("正在读取文件：", pcapfile)

    # dpkt解析数据包
    pkts = dpkt.pcap.Reader(f)
    for ts, buf in pkts:

        # 提取IP五元组
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            srcip = socket.inet_ntoa(ip.src)
            dstip = socket.inet_ntoa(ip.dst)
            protocol_index = ip.p
            if isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP):
                srcport = ip.data.sport
                dstport = ip.data.dport
        elif isinstance(eth.data, dpkt.ip6.IP6):
            ipv6 = eth.data
            srcip = socket.inet_ntop(socket.AF_INET6, ipv6.src)
            dstip = socket.inet_ntop(socket.AF_INET6, ipv6.dst)
            protocol_index = ipv6.nxt
            if isinstance(ipv6.data, dpkt.tcp.TCP) or isinstance(ipv6.data, dpkt.udp.UDP):
                srcport = ipv6.data.sport
                dstport = ipv6.data.dport

        # 提取应用名称
        pcapdroiddata = buf[-32:]
        # print('pcapdroiddata:', pcapdroiddata)
        applabel = pcapdroiddata[8: 28]
        # print('applabel:', applabel)
        appname = applabel.decode('utf-8', errors='ignore').strip(b'\x00'.decode()).replace(' ', '')
        # print('appname:', appname)
        if appname in target_applabel.keys():
            label = target_applabel[appname]
        else:
            # 其他类:
            # label = len(targetname) - 1
            label = len(targetname) - 1
        if srcip < dstip:
            key = srcip + ',' + str(srcport) + ',' + dstip + ',' + str(dstport) + ',' + str(protocol_index)
        else:
            key = dstip + ',' + str(dstport) + ',' + srcip + ',' + str(srcport) + ',' + str(protocol_index)

            # 只取同一条流的第一条流
            if key not in flow_unique and srcip not in IP_DNS_DIRECT.keys() and dstip not in IP_DNS_DIRECT.keys():
                flow_unique.add(key)
                # 一条流记录[五元组, 真实标签]
                flowinpcap.append([key, label])

    f.close()
    print("文件中流量条数：", len(flowinpcap))
    return flowinpcap


def createfrompair(predfile):
    with open(predfile, 'r') as file:
        lines = file.readlines()

    prediction_flow = []

    for i in range(0, len(lines), 2):
        # 读取特征行
        feature_line = lines[i]
        features = feature_line.split(',')[5:]
        features = ','.join(features)

        # 读取预测流
        tuple_line = lines[i + 1]
        srcip, srcport, dstip, dstport, protocol_index, label, confidence = tuple_line.strip().split(',')
        label = label.split(':')[-1].strip()
        protocol_index = protocol_index.split('.')[0]
        dstport = dstport.split('.')[0]

        if srcip not in IP_DNS_DIRECT.keys() and dstip not in IP_DNS_DIRECT.keys():
            if srcip < dstip:
                key = srcip + ',' + srcport + ',' + dstip + ',' + dstport + ',' + protocol_index
            else:
                key = dstip + ',' + dstport + ',' + srcip + ',' + srcport + ',' + protocol_index

            prediction_flow.append([key, int(label), float(confidence), features])
    return prediction_flow


# 读取路由器的预测结果
def createfromtext(textfile):
    flowintext = []
    f = open(textfile, mode='r')
    print("正在读取文件：", textfile)
    for line in f:
        # 逗号分割，分别获取对应值
        srcip, srcport, dstip, dstport, protocol_index, label, confidence = line.strip().split(',')
        label = label.split(':')[-1].strip()
        protocol_index = protocol_index.split('.')[0]
        dstport = dstport.split('.')[0]

        if srcip < dstip:
            key = srcip + ',' + srcport + ',' + dstip + ',' + dstport + ',' + protocol_index
        else:
            key = dstip + ',' + dstport + ',' + srcip + ',' + srcport + ',' + protocol_index

        # 一条流记录[五元组, 预测标签, 置信度]
        flowintext.append([key, int(label), float(confidence)])

    f.close()
    return flowintext


def preprocesstext(pairfile, predictfile):
    with open(pairfile, 'r') as file:
        lines = file.readlines()

    with open(predictfile, 'w') as file:
        for line in lines:
            if 'In recvTcpMsg, func readn() err peer closed:' not in line and 'Waiting for client connection.' not in line:
                file.write(line)


def threshold09_hist(true_thresholds, false_thresholds):
    plt.figure()
    plt.hist([true_thresholds, false_thresholds], bins=10, label=['predict_true', 'predict_false'])
    plt.xlabel('Threshold')
    plt.ylabel('Frequency')
    plt.title('Threshold Distribution Histogram')
    plt.xticks(np.arange(0.9, 1.01, 0.01))
    plt.legend()
    # plt.show()
    plt.savefig('./picture/0.9-1_histogram.png', dpi=500)


def threshold_hist(true_thresholds, false_thresholds):
    plt.figure()
    plt.hist([true_thresholds, false_thresholds], bins=10, label=['predict_true', 'predict_false'])
    plt.xlabel('Threshold')
    plt.ylabel('Frequency')
    plt.title('Threshold Distribution Histogram')
    plt.legend()
    # plt.show()
    plt.savefig('./picture/0-1_histogram.png', dpi=500)


def ratios_confusion_matrix(cm_ratios, classes):
    plt.figure(figsize=(24, 24))
    # 绘制混淆矩阵热力图
    sns.heatmap(cm_ratios, annot=True, cmap='Greens', fmt='.2f', square=True, xticklabels=True, yticklabels=True)
    plt.title('Confusion Matrix')
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = 'simhei'
    # 设置刻度标签
    tick_marks = np.arange(len(classes)) + 0.5
    plt.xticks(tick_marks, classes, fontsize=18, rotation=15)
    plt.yticks(tick_marks, classes, fontsize=18, rotation=15)
    # 设置轴标签
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.show()
    plt.savefig('./picture/confusion_matrix.png', dpi=500)


def max_mispredict(cm):
    np.fill_diagonal(cm, 0)
    index = np.unravel_index(np.argmax(cm), cm.shape)
    # 输出误识别率最高的流+特征
    mc_true = index[0]
    mc_pred = index[1]
    max_confusion = []
    mc_pred_feature = []
    mc_true_feature = []
    for pcf in pcapcatch_flow:
        # pcf = [五元组, 真实标签]
        # 遍历预测流
        for pf in prediction_flow:
            # pf = [五元组, 预测标签, 置信度, 特征]
            if pf[0] == pcf[0] and pf[1] == mc_pred and pcf[1] == mc_true:
                max_confusion.append([pf[3]])
            if pf[0] == pcf[0] and pf[1] == mc_pred and pcf[1] == mc_pred:
                mc_pred_feature.append([pf[3]])
            if pf[0] == pcf[0] and pf[1] == mc_true and pcf[1] == mc_true:
                mc_true_feature.append([pf[3]])

    localtime = time.strftime('%Y%m%d%H%M%S', time.localtime())
    mc_fpath = './output/' + targetname[mc_true] + '误识别为' + targetname[mc_pred] + '_' + localtime + '.csv'
    mc_pred_fpath = './output/' + targetname[mc_pred] + '_' + localtime + '.csv'
    mc_true_fpath = './output/' + targetname[mc_true] + '_' + localtime + '.csv'
    writelist(max_confusion, mc_fpath)
    print('正在写入文件：', mc_fpath)
    writelist(mc_pred_feature, mc_pred_fpath)
    print('正在写入文件：', mc_pred_fpath)
    writelist(mc_true_feature, mc_true_fpath)
    print('正在写入文件：', mc_true_fpath)


def max_truepredict(cm):
    index = np.unravel_index(np.argmax(cm), cm.shape)
    mc_true = index[0]
    mc_pred = index[1]
    max_confusion = []
    for pcf in pcapcatch_flow:
        # pcf = [五元组, 真实标签]
        # 遍历预测流
        for pf in prediction_flow:
            # pf = [五元组, 预测标签, 置信度, 特征]
            if pf[0] == pcf[0] and pf[1] == mc_pred and pcf[1] == mc_true:
                max_confusion.append([pf[3]])
    localtime = time.strftime('%Y%m%d%H%M%S', time.localtime())
    mc_fpath = './output/' + targetname[mc_pred] + '_正确识别率最高_' + localtime + '.csv'
    writelist(max_confusion, mc_fpath)
    print('正在写入文件：', mc_fpath)


def readreport(report):
    labels = []
    precisions = []
    recalls = []
    f1_scores = []
    lines = report.split('\n')
    for line in lines[2: (len(lines) - 5)]:
        line = line.split()
        labels.append(line[0])
        precisions.append(float(line[1]))
        recalls.append(float(line[2]))
        f1_scores.append(float(line[3]))
    accuracy, support = lines[-4].strip().split()[1:]
    return labels, precisions, recalls, f1_scores, float(accuracy), int(support)


def report_bar(report):
    # report结果
    labels, precisions, recalls, f1_scores = readreport(report)[0: 4]
    plt.figure(figsize=(16, 8))
    # 设置柱子的宽度
    bar_width = 0.25
    # 设置柱子的位置
    bar_position1 = np.arange(len(labels))
    bar_position2 = [x + bar_width for x in bar_position1]
    bar_position3 = [x + 2 * bar_width for x in bar_position1]
    # 绘制柱状图
    plt.bar(bar_position1, precisions, width=bar_width, label='precision')
    plt.bar(bar_position2, recalls, width=bar_width, label='recall')
    plt.bar(bar_position3, f1_scores, width=bar_width, label='f1_score')
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = 'simhei'
    # 设置横纵轴标签
    plt.title('Classification Report Histogram')
    plt.xticks(bar_position2, labels, fontsize=13, rotation=15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=13)
    # 显示图例
    plt.legend()
    # plt.show()
    plt.savefig('./picture/report_bar.png', dpi=500)


def report_pie(report, all):
    accuracy, support = readreport(report)[-2:]
    labels = ['>threshold and True', '>threshold and False', '<threshold']
    true_count = round(support * accuracy)
    false_count = support - true_count
    sizes = [true_count, false_count, (all - support)]
    explode = (0, 0, 0)
    plt.figure()
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')
    # plt.show()
    plt.savefig('./picture/report_piechart.png', dpi=500)


def writelist(li, listpath):
    f = open(listpath, mode='a')
    for l in li:
        if not isinstance(l, str):
            l = ','.join(l)
        f.write(l)
    f.close()


# 5app
# target_applabel = {'王者荣耀': 0, '爱奇艺': 1, 'QQ音乐': 2, '抖音': 3}
# targetname = ['王者荣耀', '爱奇艺', 'QQ音乐', '抖音', '其他']

# # 10app
# target_applabel = {'爱奇艺': 0, '金铲铲之战': 1, '哔哩哔哩': 2, '中国大学MOOC': 3, '虎牙直播': 4, '百度贴吧': 5,
#                    'QQ音乐': 6}
# targetname = ['爱奇艺', '金铲铲之战', '哔哩哔哩', '中国大学MOOC', '虎牙直播', '百度贴吧','QQ音乐','Background']

# 23app
# target_applabel = {'王者荣耀': 0, '爱奇艺': 1, 'QQ音乐': 2, '抖音': 3, '和平精英': 4, '金铲铲之战': 5, '哔哩哔哩': 6, '腾讯会议': 7, '中国大学MOOC': 8, '原神': 9, 'QQ飞车': 10, '网易会议': 11, '香肠派对': 12, '百度贴吧': 13, '虎牙直播': 14, '斗鱼': 15, '腾讯视频': 16, '火影忍者': 17, '知乎': 18, '蛋仔派对': 19, '微博': 20, '英雄联盟手游': 21}
# targetname = ['王者荣耀', '爱奇艺', 'QQ音乐', '抖音', '和平精英', '金铲铲之战', '哔哩哔哩', '腾讯会议', '中国大学MOOC', '原神', 'QQ飞车', '网易会议', '香肠派对', '百度贴吧', '虎牙直播', '斗鱼', '腾讯视频', '火影忍者', '知乎', '蛋仔派对', '微博', '英雄联盟手游', 'VR', '背景']

# 14app
target_applabel = {'QQ音乐': 0, '百度贴吧': 1, '腾讯会议': 2, '虎牙直播': 3, '王者荣耀': 4, '腾讯视频': 5,
                   '哔哩哔哩': 6, '抖音': 7, '中国大学MOOC': 8, '金铲铲之战': 9, '爱奇艺': 10, '香肠派对': 11, '和平精英': 12, '优酷视频': 13}
targetname = ['QQ音乐', '百度贴吧','腾讯会议','虎牙直播','王者荣耀','腾讯视频','哔哩哔哩','抖音','中国大学MOOC','金铲铲之战','爱奇艺','香肠派对','和平精英','优酷视频','背景']
# 带标记流量文件位置
pcappath = './apppcap/'
# 预测结果文件位置
predictionpath = './prediction.txt'
# 置信度阈值
threshold = 0.98

# 使用的model
# model = 'netsniff'
model = 'routers‘dpisniff'

if __name__ == '__main__':
    # pcapdroid文件目录
    pcappath = './apppcap/'
    # 预测结果+特征
    #pairfile = './raw.csv'
    predictionpath = './prediction.txt'
    vrip = '192.168.2.5'

    pcappathlist = walkFile(pcappath)

    # 真实流
    pcapcatch_flow = []
    # 预测流
    prediction_flow = []

    for pcapfile in pcappathlist:
        ipdnsdirect(pcapfile)

    for pcapfile in pcappathlist:
        # 读取真实流pcap
        pcapcatch_flow += createfrompcap(pcapfile)

    # 读取预测流text
    #preprocesstext(pairfile, predictionpath)
    # prediction_flow = createfrompair(predictionpath)
    prediction_flow = createfromtext(predictionpath)

    print('pcapdroid抓取真实流数: ', len(pcapcatch_flow))
    print('预测流数: ', len(prediction_flow))

    # flag = False流记录数
    count = 0
    # 真实标签
    true_label = []
    # 预测标签
    predict_label = []
    # 低于阈值的流
    low_threshold = []
    # 阈值取值
    true_thresholds = []
    false_thresholds = []
    true_thresholds09 = []
    false_thresholds09 = []

    # 遍历pcapdroid真实流
    for pcf in pcapcatch_flow:
        # pcf = [五元组, 真实标签]
        # 遍历预测流
        for pf in prediction_flow:
            # pf = [五元组, 预测标签, 置信度]
            # 如果五元组匹配成功,则说明两个都有
            if pf[0] == pcf[0]:
                count += 1
                if pf[2] >= threshold:
                    true_label.append(pcf[1])
                    predict_label.append(pf[1])
                else:
                    # 低于阈值的流
                    low_threshold.append([pcf[0], str(pcf[1]), str(pf[1]), str(pf[2])])

                #  直方图的阈值参数
                if pf[2] >= 0.9:
                    if pf[1] == pcf[1]:
                        true_thresholds09.append(pf[2])
                    else:
                        false_thresholds09.append(pf[2])
                if pf[1] == pcf[1]:
                    true_thresholds.append(pf[2])
                else:
                    false_thresholds.append(pf[2])

    print('pcapdroid有,dpisniff没有的流记录数: ', len(pcapcatch_flow) - count)
    print('dpisniff有,pcapdroid没有的流记录数: ', len(prediction_flow) - count)

    # # 统计VR
    # vr_no = len(targetname) - 2
    # for pf in prediction_flow:
    #     if vrip in pf[0]:
    #         true_label.append(vr_no)
    #         predict_label.append(pf[1])
    #         if pf[2] >= 0.9 and pf[1] == vr_no:
    #             true_thresholds09.append(pf[2])
    #         elif pf[2] >= 0.9 and pf[1] != vr_no:
    #             false_thresholds09.append(pf[2])
    #         elif pf[1] == vr_no:
    #             true_thresholds.append(pf[2])
    #         elif pf[1] != vr_no:
    #             false_thresholds.append(pf[2])
    #
    appflow_count = len(true_thresholds) + len(false_thresholds)

    print("true_label", set(true_label))
    print("predict_label", set(predict_label))
    # writelist(low_threshold, './output/low_threshold.csv')
    # print('正在写入文件：./output/low_threshold.csv')
    # 计算指标
    report = classification_report(true_label, predict_label, target_names=targetname)
    print(report)

    f = open('./output/indicator_results.txt', mode='a')
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n'
    f.write(t)
    f.write(str(target_applabel) + '\n')
    f.write('描述：使用的模型是 ' + model +
            '; 置信度阈值是' + str(threshold) +
            '; pcapdroid抓取真实流数是 ' + str(len(pcapcatch_flow)) +
            '; 预测流数是 ' + str(len(prediction_flow)) +
            '; pcapdroid有，dpisniff没有的流记录数是 ' + str(count) + '\n')
    f.write(report)
    f.close()

    #     report = '              precision    recall  f1-score   support\n\n \
    #         王者荣耀       1.00      0.85      0.92        60\n \
    #          爱奇艺       0.86      0.98      0.92       193\n \
    #         QQ音乐       0.99      0.99      0.99       785\n \
    #           抖音       0.96      0.93      0.95       340\n \
    #         和平精英       0.97      0.87      0.91        67\n \
    #        金铲铲之战       0.99      1.00      0.99      1337\n \
    #         哔哩哔哩       0.98      0.96      0.97       515\n \
    #         腾讯会议       0.92      0.89      0.91       135\n \
    #     中国大学MOOC       0.96      0.98      0.97       312\n \
    #           背景       0.91      0.89      0.90       199\n\n \
    #     accuracy                           0.97      3943\n \
    #    macro avg       0.95      0.93      0.94      3943\n \
    # weighted avg       0.97      0.97      0.97      3943\n'

    # 绘制阈值的直方图
    threshold09_hist(true_thresholds09, false_thresholds09)
    threshold_hist(true_thresholds, false_thresholds)
    # 绘制report结果直方图
    report_bar(report)
    # 绘制饼图
    report_pie(report, appflow_count)
    # 计算混淆矩阵
    cm = confusion_matrix(true_label, predict_label)
    # 每一类真实标签的总数
    class_totals = np.sum(cm, axis=1)
    cm_ratios = cm / class_totals[:, np.newaxis]
    ratios_confusion_matrix(cm_ratios, targetname)

    # # 输出正确识别率最高的流+特征
    # max_truepredict(cm_ratios)
    # # 输出误识别率最高的流+特征
    # max_mispredict(cm_ratios)
