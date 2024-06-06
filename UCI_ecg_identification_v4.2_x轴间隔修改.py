
import io, os,sys
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
##from feature_extraction import computeBpm
from scipy.signal import find_peaks, butter, filtfilt
from pylab import xticks, yticks, np
from utils import find_largest_smaller_than_or_equal, detect_peaks
# from feature_extraction import ppg_show

import matplotlib
##matplotlib.use("Agg")

def computeBpm(data, sampling_rate):
    N = len(data)
    # print(N)
    # print('The sampling_rate is: ', sampling_rate)
    data_fft = np.fft.fft(data)[:N // 2]  # 只取正频率部分
    magnitudes = np.abs(data_fft)
    data_freq = np.fft.fftfreq(len(data))

    # 可视化频谱
    # plt.plot(data_freq, magnitudes)
    test_index = np.fft.fftfreq(N)[:N // 2] * sampling_rate
    # plt.plot(test_index, magnitudes)
    # plt.title('Frequency Spectrum')
    # plt.show()

    # sampling_rate = 125
    ppg2 = (data - np.min(data))
    # f, Pxx_den = signal.periodogram(ppg2, 125)
    f, Pxx_den = signal.periodogram(ppg2, sampling_rate)
    # plt.plot(f, Pxx_den)
    # plt.show()

    picks = int(len(data) / sampling_rate)
    picks_start = int(0.75 * picks)
    picks_end = int(4 * picks)
    # print('1Hz # of data points', picks)
    # print(picks_start)
    # print(picks_end)

    peaks, _ = find_peaks(Pxx_den[picks_start:4 * picks], distance=0.7 * picks)
    # print('length of dengram is ', len(Pxx_den))
    peaks2 = peaks + picks_start
    # print(peaks2)
    # print(Pxx_den[peaks2])
    # print(np.argmax(Pxx_den[peaks2]))
    # print(f[peaks2[0]])
    # print(f[peaks2[1]])
    if abs(peaks2[1] - 2 * peaks2[0]) < 0.03 * picks:
        max_index = peaks2[0]
    elif peaks2[0]> picks:
        max_index = peaks2[0]
    elif Pxx_den[peaks2[1]] > 5 * Pxx_den[peaks2[0]]:
        max_index = peaks2[1]
    else:
        max_index = peaks2[np.argmax(Pxx_den[peaks2[0:2]])]
    # print(max_index)
    # freq = peaks[max_index]
    bpm = (max_index) / N * sampling_rate * 60  # 将频率转换为心率（bpm）
    # print('the freq is: ', freq)
    print('The bpm is: ', bpm)

    # print(max_index, magnitudes[max_index+shift])
    # print(len(test_index))

    return bpm


def period_division(signal, mypeaks, myvalleys):
    # 分割出ppg周期，不完整的周期丢弃
    # input: ppg_signal ppg信号段, ppg_peaks , ppg_valleys

    # peak 和valley升序合并
    peaks_and_valleys = [(num, 1) for num in mypeaks] + [(num, 2) for num in myvalleys]
    peaks_and_valleys.sort(key=lambda x: x[0])

    # 存在出现连续的波峰或波谷问题（存在重搏波），遇到连续波峰取高的，连续波谷取右侧的
    peaks_and_valleys_new = []
    if len(peaks_and_valleys):
        peaks_and_valleys_new.append(peaks_and_valleys[0])

    for pv in peaks_and_valleys:
        if pv[1] != peaks_and_valleys_new[-1][1]:
            peaks_and_valleys_new.append(pv)
        elif pv[1] == 1 and signal[pv[0]] >= signal[peaks_and_valleys_new[-1][0]]:
            peaks_and_valleys_new.pop()
            peaks_and_valleys_new.append(pv)
        elif pv[1] == 2:
            peaks_and_valleys_new.pop()
            peaks_and_valleys_new.append(pv)
            # print("pv", pv)

    peaks = [pv[0] for pv in peaks_and_valleys_new if pv[1] == 1]
    valleys = [pv[0] for pv in peaks_and_valleys_new if pv[1] == 2]

    # 保证数据第一个是波谷，保证周期完整
    if len(peaks_and_valleys_new) and peaks_and_valleys_new[0][1] == 1:
        peaks_and_valleys_new = peaks_and_valleys_new[1:]

    # 保证数据最后一个是波谷，保证周期完整
    if len(peaks_and_valleys_new) and peaks_and_valleys_new[-1][1] == 1:
        peaks_and_valleys_new = peaks_and_valleys_new[:-1]

    result = []
    if len(peaks_and_valleys_new) >= 3:
        # print("nums", (len(peaks_and_valleys_new) - 1)/2)
        nums = int((len(peaks_and_valleys_new) - 1) / 2)
        for i in range(0, nums):
            j = i * 2 + 1
            start = peaks_and_valleys_new[j - 1][0]
            end = peaks_and_valleys_new[j + 1][0]
            peak = peaks_and_valleys_new[j][0]
            result.append([start, end, peak])

    return result, peaks, valleys


def abnormal_Identifier(myPeriods):
    time_Abnormal = []
    tmp_peaks = myPeriods.iloc[:,2]
    medianHeight = np.median(myPeriods.iloc[:,4])
    pLength = np.diff(tmp_peaks)
    rr_interval = np.array(pLength)

    rr_mean = np.median(rr_interval)
    test_std = np.std(rr_interval)

    length_index = len(myPeriods)

    for i in range(0,length_index-1):
        if i < 20:
            test_mean = rr_mean
        else:
            test_mean = np.median(rr_interval[i-20:i])
        # print('the mean period is ', myPeriods.iloc[i,:], test_mean, 2*test_std, rr_interval[i])
        time_threshold = max(0.2 * test_mean, 2 * test_std)
        start = myPeriods.iloc[i,2]  # from the peak to the end, locate valley2
        end = myPeriods.iloc[i,1]
        valley2 = myPeriods.iloc[i,5]
        a = abs(test_mean - rr_interval[i]) > time_threshold or (valley2-myPeriods.iloc[i,0])>15 or (myPeriods.iloc[i,4])<0.5*medianHeight
        b = [abs(test_mean - rr_interval[i]) > time_threshold, (valley2-myPeriods.iloc[i,0])>15, (myPeriods.iloc[i,4])<0.5*medianHeight]

        # print('a 1st is ', abs(test_mean - rr_interval[i]) > time_threshold)
        # print('a 1st is ', (valley2-myPeriods.iloc[i,0])>10)
        # print('a 1st is ', (myPeriods.iloc[i,4])<0.5*medianHeight)


        # print('valley2 is', myPeriods.iloc[i,0], start, valley2, (valley2-myPeriods.iloc[i,0])>15)
        if a==True:
            # print('the check boolean is ', myPeriods.iloc[i,0], b)
            test = pd.Series(myPeriods.iloc[i, :])
            test[6] = i + 1
            time_Abnormal.append(list(test))

    return time_Abnormal

def ecg_PVC_Selection(signal, fs):
    y = signal*10
    b, a = butter(2, [1.4 / 125, 10 / 125], 'bandpass')
    ecg = filtfilt(b, a, y)
    bpm = computeBpm(ecg,125)
    mydistance = 2*60/bpm*125

    #获取 peaks和valleys
    tbeat = 60/bpm  #心跳一次的平均时间
    mpd2 = int(tbeat*fs*0.4)
    # print('mpd', mpd2)

    temp_min = np.min(y)
    temp_max = np.max(y)
    mph2 = 0.05*(temp_max - temp_min)
    signal_peak = y - temp_min

    x = signal_peak
    peaks2 = detect_peaks(x, mpd=mpd2, mph=mph2)

    medianHeight = np.median(x[peaks2])
    mph = 0.4*medianHeight
    mpd = mpd2
    peaks = detect_peaks(x, mpd=mpd, mph=mph)

    mmpd = int(tbeat*fs*0.1)

    signal_valley = temp_max - x

    # valleys = detect_peaks(ppg3, valley=True, edge='falling', mpd = mmpd, mph = mph)
    valleys = detect_peaks(signal_valley, mpd = 1, edge='rising')
    # print('valleys\n', valleys)

    signal_period, peaks, valleys = period_division(x, peaks, valleys)

    periods = pd.DataFrame(signal_period)
    # print('periods is ', periods)
    factor = 1
    myPeriods = []
    tmp = []
    tmp2 = []
    # print('the length of periods', len(periods))
    for i in range(0, len(periods)):
        start = periods.iloc[i,2]  # from the peak to the end, locate valley2
        end = start + 30
        valley2 = np.argmin(x[start:end]) + start
        tmp = [int(periods.iloc[i,0]/factor), int(periods.iloc[i,1]/factor), int(periods.iloc[i,2]/factor), periods.iloc[i,1]-periods.iloc[i,0], x[periods.iloc[i,2]] - x[periods.iloc[i,0]], valley2]
        tmp2.append(tmp)
    myPeriods = pd.DataFrame(tmp2)
    # print('myPeriods is ', myPeriods)

    # print('myPeriods is: ', myPeriods)
    # print('the mean and std of rr intervals', np.mean(myPeriods.iloc[:,3]), np.std(myPeriods.iloc[:,3]))
    # print('the mean and std of peak heights', np.mean(myPeriods.iloc[:,4]), np.std(myPeriods.iloc[:,4]))

    pLength = np.diff(myPeriods.iloc[:,2])
    rr_interval = np.array(pLength)
    # print('length of rr intervals', rr_interval)

    abnormals = abnormal_Identifier(myPeriods)
    # print('the abnormals are ', abnormals)
    test4 = []

    if len(abnormals)>0:
        temp = pd.DataFrame(abnormals).astype(int)
        test4 = temp.iloc[:,2]
        amplifier = 1
        endIndex = len(x)
        # test4.sort()
        if len(test4)>0:
            cnt = 0
            # if cnt > max_figure_num:
            #     with open('skip.txt', 'a+', encoding='utf-8') as file:
            #         file.write(str(filename) + ' ')
            # else:
            #     with open('notskip.txt', 'a+', encoding='utf-8') as file:
            #         file.write(str(filename) + ' ')
            print(f"本次运行应该保存{len(temp)}张图片!")
            for i in range(len(temp)):
                index = i
                peak = temp.iloc[index,2]
                start = max(0, peak-400)
                end = min(peak+400, endIndex)
                start_abnormal = max(0, peak-150)
                end_abnormal = min(peak+150, endIndex)
                valley1 = temp.iloc[index,0]
                valley2 = temp.iloc[index,5]
                myXLabel = list(range(int(start*amplifier), int(end*amplifier), amplifier))
                myX2Label = list(range(int(start_abnormal*amplifier), int(end_abnormal*amplifier), amplifier))

                plt.figure(figsize=(20,10.5))
                # plt.plot(range(start,peak+400), x[start:peak+400]*20-10, 'b', range(start,peak+400), y[start:peak+400]*10, 'y',range(start,peak+400), (z[start:peak+400]-zMean)/2 +40, 'black')
                plt.plot(myXLabel, x[start:end]*20-10, 'b')
                plt.plot(peak*amplifier, x[peak]*20-10, 'o')
                plt.plot(valley1*amplifier, x[valley1]*20-10, 'r*')
                plt.plot(valley2*amplifier, x[valley2]*20-10, 'b*')
                
##                调整间隔
                myTicks = math.floor((end-start) * amplifier / 25) + 1
                end = start + (myTicks-1)* 25
                # print(start,end,amplifier,myTicks)

                # xticks(np.linspace(0, endIndex, myTicks, endpoint=True))
                xticks(np.linspace(start*amplifier, end*amplifier, myTicks, endpoint=True))
                plt.plot(myX2Label, x[start_abnormal:end_abnormal]*20-10, "r")
                plt.title(str(filename) + '.csv' + ': The ECG Lead I signals at time ' + f"{peak*amplifier}" + ' :original index is ' + f"{peak}")
                # plt.legend(["ECG", "PPG", "SBP", "PVC candidate"], loc="upper right")
                plt.legend(["ECG", "Abnormal candidate"], loc="upper right")
                plt.grid(linestyle = '-.')
                plt.ylim(-250, 500)
                
# ##                字体不遮挡
#                 plt.xticks(rotation=45)
                plt.xticks(rotation=0)
                # 调整窗口在屏幕上弹出的位置
##                mngr = plt.get_current_fig_manager()      
##                mngr.window.wm_geometry("+0+0")

                # 保存图片与展示
                plt.savefig(r'E:\dl_data\figure/{}_{}.png'.format(filename, peak))
##                plt.show()
                cnt += 1
        else:
            test4=[]
        # print(peaks[test4][0,:])
    print(f"本次运行保存了{cnt}张图片")
    return test4

## 保存图片超过限制时使用下行代码
matplotlib.use("Agg")
if __name__ == '__main__':
    # 将以下路径名改为存放数据文件的文件夹所在路径
    root_path = r'E:\dl_data\data\2501-3000'
    # root_path = r'../../UCI DataSet/Part_4'
    pvc = []
    start = 2501
    end = 3001
    notskip = []
    with open('notskip.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            ls = line.strip().split()
            for c in ls:
                if c.isdigit():
                    notskip.append(int(c))
    # print(len(notskip))
    # print(notskip)
    misslist = []
    # for i in range(start,end):
    for i in notskip[150:]:
        if i<2958:
            continue
        print('Dealing with the ', i, 'th file' )
        filename = i
        file_path = os.path.join(root_path, str(filename) + '.csv')
        df = pd.read_csv(file_path, header=None, names=['ppg', 'sbp', 'ecg'])
        myLength = len(df)
        # print('The processed csv file is: ', file_path)
        x = df['ecg']
        # y = df['ppg']
        # z = df['sbp']
        # zMean = np.mean(z)

        temp = ecg_PVC_Selection(x, 125)
        pvc.append(list(temp))

        # plt.figure(figsize=(15,8))
        # plt.plot(x)
        # plt.title('The ECG Lead I signals of Patient ' + str(filename) + ' from ' + str(filename) + '.csv')
        # plt.show()

    # print(pvc)

    # pvcPD = pd.DataFrame(data = pvc, index=range(start,end))
    # pvcPD.to_csv('candidates.csv')

