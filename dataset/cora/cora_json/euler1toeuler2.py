import json
from tqdm import tqdm
def list2str(x):
    return str(int(x))
if __name__=='__main__':
    tmp = []
    for line in tqdm(open('./data.json','r')):
        tmp.append(json.loads(line))
    content = open("cora_euler1.content", "w")

    for i in tqdm(tmp):
        tmp_list =[str(i['node_id'])] 
        tmp_list.extend(list(map(list2str,i['float_feature']['1'])))
        tmp_list.append(str(i['float_feature']['0'].index(1)))
        content.writelines('\t'.join(tmp_list))
        content.writelines('\n')
    content.close()
    cite_tmp = []
    for i in tqdm(tmp):
        x = str(i['node_id'])
        for j in i['neighbor']['0'].keys():
            if [x,j] not in cite_tmp:
                cite_tmp.append([x,j])
    cites = open("cora_euler1.cites", "w")
    for i in tqdm(cite_tmp):
        cites.writelines('\t'.join(i))
        cites.writelines('\n')
    cites.close()