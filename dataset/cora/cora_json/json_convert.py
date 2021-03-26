import json
from tqdm import tqdm

def convert_euler1_line_to_euler2_line(tmp):
    tmp_node_buf={}
    tmp_node_buf['id']=tmp['node_id']
    tmp_node_buf['type']=tmp['node_type']
    tmp_node_buf['weight']=tmp['node_weight']
    tmp_node_buf['features']=[]

    for i in tmp['uint64_feature'].keys():
        feat_tmp={}
        feat_tmp['name'] = i
        feat_tmp['type'] = 'sparse'
        feat_tmp['value']=tmp['uint64_feature'][i]
        tmp_node_buf['features'].append(feat_tmp)
    for i in tmp['float_feature'].keys():
        feat_tmp={}
        feat_tmp['name'] = i
        feat_tmp['type'] = 'dense'
        feat_tmp['value']=tmp['float_feature'][i]
        tmp_node_buf['features'].append(feat_tmp)
    for i in tmp['binary_feature'].keys():
        feat_tmp={}
        feat_tmp['name'] = i
        feat_tmp['type'] = 'binary'
        feat_tmp['value']=tmp['binary_feature'][i]
        tmp_node_buf['features'].append(feat_tmp)
    euler2json['nodes'].append(tmp_node_buf)
    if tmp_node_buf['type'] == 1:
        out_test.write(str(tmp_node_buf['id']) + "\n")
    for i in tmp['edge']:
        edge_tmp={}
        edge_tmp['src'] = i['src_id'] 
        edge_tmp['dst'] = i['dst_id']
        edge_tmp['type']= i['edge_type']
        edge_tmp['weight'] = i['weight']
        edge_tmp['features'] = []

        for j in i['uint64_feature'].keys():
            feat_tmp={}
            feat_tmp['name'] = j
            feat_tmp['type'] = 'sparse'
            feat_tmp['value']=i['uint64_feature'][j]
            edge_tmp['features'].append(feat_tmp)
        for j in i['float_feature'].keys():
            feat_tmp={}
            feat_tmp['name'] = j
            feat_tmp['type'] = 'dense'
            feat_tmp['value']=i['float_feature'][j]
            edge_tmp['features'].append(feat_tmp)
        for j in i['binary_feature'].keys():
            feat_tmp={}
            feat_tmp['name'] = j
            feat_tmp['type'] = 'binary'
            feat_tmp['value']=i['binary_feature'][j]
            edge_tmp['features'].append(feat_tmp)
    euler2json['edges'].append(edge_tmp)
    
euler2json={'nodes':[],'edges':[]}
out_test = open('./json_convert/BGN_test.id', 'a+')
out = open('./json_convert/data.json', 'a+') 

for line in tqdm(open('./data.json','r')):
    convert_euler1_line_to_euler2_line(json.loads(line))
out.write(json.dumps(euler2json))
out.close()
out_test.close()