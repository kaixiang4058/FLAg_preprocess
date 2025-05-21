import os
import json
import argparse
import random
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pklroot_label', default="./dataset/pkl/MRCPS_p512_s384_label_level0/")
    parser.add_argument('--pklroot_unlabel', default="./dataset/pkl/MRCPS_p512_s512_unlabel_level0/")
    parser.add_argument('--saveroot', default="./dataset/")
    args = parser.parse_args()
    
    label_list=[]
    tmp_list = os.listdir(args.pklroot_label)
    for tmp in tmp_list:
        if tmp.split('.')[-1] == 'pkl':
            label_list.append(tmp.split('.')[0])


    unlabel_list=[]
    tmp_list = os.listdir(args.pklroot_unlabel)
    for tmp in tmp_list:
        if tmp.split('.')[-1] == 'pkl':
            unlabel_list.append(tmp.split('.')[0])

    order=['train','valid','test']
    ratio=[7,1,2]   #label data ratio each stage
    data_dict = {'train':{},'valid':{},'test':{}}
    
    label_num = len(label_list)
    print(f'label total number: {label_num}')
    print(f'unlabel total number: {len(unlabel_list)}')

    #label set
    if label_num<3:
        for dk in data_dict.keys():
            data_dict[dk]['label']=label_list
    else:
        rand_index = list(range(label_num))
        random.shuffle(rand_index)  #radnom
        for i in range(len(order)):
            data_dict[order[i]]['label'] = [label_list[rand_index[i]]]
        
        accumulate = 0
        ratio_accumulate=[]
        for v in ratio:
           accumulate+=v
           ratio_accumulate.append(accumulate)
        for i in range(len(order)):
            now_i = round(ratio_accumulate[i]/accumulate*(label_num-3))
            new_list = []
            if i == 0:
                pre_i = 0
            else:
                pre_i = round(ratio_accumulate[i-1]/accumulate*(label_num-3))
             
            for index in rand_index[pre_i+len(order):now_i+len(order)]:
                new_list.append(label_list[index])
           
            data_dict[order[i]]['label'] += new_list
            print(f'label_{order[i]} number: {len(new_list)+1}')

    #unlabel set
    data_dict['train']['unlabel']=unlabel_list


    # print(data_dict)

    save_path = os.path.join(args.saveroot, 'datalist.json')
    with open(save_path, 'w') as jw:
        json.dump(data_dict, jw)