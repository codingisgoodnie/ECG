# -*- coding: utf-8 -*-            
# @Author : JunhanXu
# @Time : 2024/6/6 13:36
from collections import defaultdict
import os
def clean_figure(dir_path):
    """
    :param dir_path: 图片地址
    :return: 删除冗余的图片
    """
    person_to_figure = defaultdict(list) # 每个人(csv)生成的图片
    figure_list = os.listdir(dir_path) # 所有的图片文件
    for name in figure_list: # 所有的文件
        if '.png' in name: # 对于每个图片文件
            # person: csv time: 时间点
            person, time = int(name.split('.')[0].split('_')[0]), int(name.split('.')[0].split('_')[1])
            # 对每个人的时间点图片建立一个map: map{person:times} type: person:int times:list
            person_to_figure[person].append(time)
    for k in person_to_figure.keys():
        # 对每个人的图片按照时间排序
        person_to_figure[k].sort()
    for person, times in person_to_figure.items():
        # vis记录该图片是否删除 vis[i]==1表示需要删除, vis[i]==0表示不需要删除
        vis = [0]*len(times)
        start = 0 # 从第0张图片开始
        # 如果该病人的时间点存在小于400的图片
        # 假设 12,71,245,368,385 五张图片，则它们对应的时间区间为0-12,0-71,0-245,0-368,0-385
        # 此时只需要保留最后一张图片即可覆盖之前所有图片的区间
        # start表示最后一个小于400的时间点
        if times[0] < 400:
            for i in range(len(times)):
                # 先把所有的<400的时间点置1(删除)
                if times[i] < 400:
                    vis[i] = 1
                else:
                    # 如果当前的时间点大于400,就把之前的一张图片保留
                    vis[i-1] = 0
                    start = i-1
                    break
        if times[-1] < 400:
            # 如果当前的图片时间点都小于400,那就只保留最后一张图片
            start = len(times)
            vis[-1] = 0
        print(person)
        print(vis)
        # 假设start位置的图片时间点为x，代表区间是[x-400,x+400]
        l, r = start, start+2
        while r < len(times):
            # times[r] 代表区间 {times[r]-400, times[r]+400}
            # tiems[l] 代表区间 {times[l]-400, times[l]+400}
            # 也就是说times[l]和times[r]之间保留了800的空间
            # 如果times[l]和times[r]的间隔小于600(留200冗余)
            # 那就说明times[r-l]没必要存在，删除
            if times[r]-times[l] <= 600:
                vis[r-1] = 1
                r += 1
            else:
                # 如果大于600
                # 那就从r-1开始，确保时间区间边界保留
                l = r-1
                # l,r的初始间隔为1
                r = l+2
        for i in range(len(vis)):
            if vis[i]:
                # 删除冗余图片
                delete_figure_name = str(person)+'_'+str(times[i])+'.png'
                os.remove(dir_path + os.sep + delete_figure_name)

if __name__ == "__main__":
    figure_path = r""
    # clean_figure(figure_path)
