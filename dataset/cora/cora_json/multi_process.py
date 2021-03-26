import json
import ast
import os
import time
import gc
import multiprocessing as mul
from tqdm import tqdm
path_total = './direct_convert'
part_num = '2'
path_part  = '/part_{}'.format(part_num)        
part = 32
data_path = '.'
def convert_euler1_line_to_euler2_line(tmp,euler2json):
    tmp_node_buf={}
    tmp_node_buf['id']=tmp['node_id']
    tmp_node_buf['type']=tmp['node_type']
    tmp_node_buf['weight']=tmp['node_weight']
    tmp_node_buf['features']=[]

    for i in tmp['uint64_feature'].keys():
        feat_tmp={}
        feat_tmp['name'] = str(i)
        feat_tmp['type'] = 'sparse'
        feat_tmp['value']=tmp['uint64_feature'][i]
        tmp_node_buf['features'].append(feat_tmp)
    for i in tmp['float_feature'].keys():
        feat_tmp={}
        feat_tmp['name'] = str(i)
        feat_tmp['type'] = 'dense'
        feat_tmp['value']=tmp['float_feature'][i]
        tmp_node_buf['features'].append(feat_tmp)
    for i in tmp['binary_feature'].keys():
        feat_tmp={}
        feat_tmp['name'] = str(i)
        feat_tmp['type'] = 'binary'
        feat_tmp['value']=tmp['binary_feature'][i]
        tmp_node_buf['features'].append(feat_tmp)
    euler2json['nodes'].append(tmp_node_buf)
   # if tmp_node_buf['type'] == 1:
   #    out_test.write(str(tmp_node_buf['id']) + "\n")
    for i in tmp['edge']:
        edge_tmp={}
        edge_tmp['src'] = i['src_id'] 
        edge_tmp['dst'] = i['dst_id']
        edge_tmp['type']= i['edge_type']
        edge_tmp['weight'] = i['weight']
        edge_tmp['features'] = []

        for j in i['uint64_feature'].keys():
            feat_tmp={}
            feat_tmp['name'] = str(j)
            feat_tmp['type'] = 'sparse'
            feat_tmp['value']=i['uint64_feature'][j]
            edge_tmp['features'].append(feat_tmp)
        for j in i['float_feature'].keys():
            feat_tmp={}
            feat_tmp['name'] = str(j)
            feat_tmp['type'] = 'dense'
            feat_tmp['value']=i['float_feature'][j]
            edge_tmp['features'].append(feat_tmp)
        for j in i['binary_feature'].keys():
            feat_tmp={}
            feat_tmp['name'] = str(j)
            feat_tmp['type'] = 'binary'
            feat_tmp['value']=i['binary_feature'][j]
            edge_tmp['features'].append(feat_tmp)
        euler2json['edges'].append(edge_tmp)
    return euler2json
def convert_and_save(arg_total):
    list_tmp = arg_total[0]
    #out_test = open('./direct_convert/part_0/cora_test_{}.id'.format(arg_total[1]), 'w')
    out = open(path_total+path_part+'/data_{}.json'.format(arg_total[1]), 'w') 
    euler2json={'nodes':[],'edges':[]}
    for line in list_tmp:
        line = ast.literal_eval(line)
        euler2json = convert_euler1_line_to_euler2_line(line,euler2json)
    out.write(json.dumps(euler2json))
    out.close()
    return euler2json
if __name__ == '__main__':
    start_time = time.time()
    if os.path.exists(path_total+path_part) == False:
        os.mkdir(path_total+path_part)
    total_out = open(path_total+'/data_{}_total.json'.format(part_num), 'w')
    f = open('./data.json')
    tmpList = f.readlines()
    totalList = []
    step = len(tmpList)//part
    for i in range(part):
        if i==(part-1):
            totalList.append(tmpList[step*i:])
        else:
            totalList.append(tmpList[step*i:step*(i+1)])
    pool = mul.Pool(part)
    rel = pool.map(convert_and_save,[(totalList[i],i) for i in range(part)])
    totaljson = rel[0]
    for i in tqdm(range(1,len(rel))):
        totaljson['nodes'].extend(rel[i]['nodes'])
        totaljson['edges'].extend(rel[i]['edges'])
    total_out.write(json.dumps(totaljson))
    end_time = time.time()
    print('used time: ',(end_time-start_time))
