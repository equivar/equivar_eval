import sys
import logging
import importlib.metadata
import time
import math
import numpy
import torch
from torch_geometric.loader import DataLoader
import csv
from equivar_eval.config import g_config
from equivar_eval.process import AtomsToGraphs,InMemoryDatasetUtil

SQ2_1=1.0/math.sqrt(2.0)
SQ3_1=1.0/math.sqrt(3.0)
SQ23_1=SQ2_1*SQ3_1

def write_csv(datas,path_out):
    '''Write results to a csv file
    '''
    csvwriter=csv.writer(open(path_out,'w'))
    N=datas.shape[1]-1
    csvwriter.writerow(['ids']+['prediction']*N)
    [csvwriter.writerow(_data) for _data in datas]

def _ctime(secs=None):
    return time.asctime(time.localtime(secs))

def main():
    _tstart=time.time()
    logging.basicConfig(handlers=[logging.FileHandler('equivar_eval.log'),logging.StreamHandler(sys.stdout)],
            level=logging.DEBUG,format='%(levelname)s %(message)s')
    logging.info(f'##### starting module {__name__} at {_ctime(_tstart)} #####')
    _ver=importlib.metadata.version('equivar_eval')
    logging.info(f'equivar_eval version {_ver}')

    cuda_device_count=torch.cuda.device_count()
    if cuda_device_count==0:
        device='cpu'
        logging.info(f'Running on CPU (device={device})')
    elif cuda_device_count==1:
        device='cuda'
        logging.info(f'Running on a single GPU (device={device})')
    else:
        device='cuda'
        logging.warning(f'{cuda_device_count} GPUs are present, but parallel job is not implemented yet, running on a single GPU (device={device})')

    _saved_model_path=g_config['saved_model_path']
    logging.info(f'loading model from \'{_saved_model_path}\'')
    model=torch.jit.load(_saved_model_path,map_location=torch.device(device))
    logging.info('Done')
    model.eval()
    logging.info(f'Converting input data')

    graph_max_radius=g_config['graph_max_radius'] if 'graph_max_radius' in g_config else 3.0
    num_radial=g_config['num_radial'] if 'num_radial' in g_config else 32
    edge_sh_lmax=g_config['edge_sh_lmax'] if 'edge_sh_lmax' in g_config else 2
    radial_basis=g_config['radial_basis'] if 'radial_basis' in g_config else None
    a2g=AtomsToGraphs(
            path_in=g_config['data_dir'],
            graph_max_radius=graph_max_radius,
            num_radial=num_radial,
            edge_sh_lmax=edge_sh_lmax,
            radial_basis=radial_basis,
            )
    data,slices=a2g.convert()
    dataset=InMemoryDatasetUtil(data,slices)

    logging.info('Done')
    _batch_size=g_config['batch_size'] if 'batch_size' in g_config else 10
    logging.info(f'dataset size: {len(dataset)} batch size: {_batch_size}')
    predictions=[]
    data_loader=DataLoader(
            dataset,
            batch_size=_batch_size,
            shuffle=False,
            pin_memory=True
            )
    logging.info('evaluating...')
    with torch.no_grad():
        for i,data in enumerate(data_loader):
            data=data.to(device)
            data_dict=data.to_dict()
            if '_num_nodes' in data_dict:
                # torch geometric saves value at this key as a list, while jit expects a tensor
                data_dict['_num_nodes']=torch.tensor(data_dict['_num_nodes'],dtype=data_dict['_num_nodes'][0].dtype)
            out=model(data_dict)
            _o=out.cpu().numpy()
            predictions=_o if i==0 else numpy.vstack((predictions,_o))
            ids_temp=[_id_atom for _id_atom in data.structure_id.cpu().numpy()]
            if i==0:
                ids=ids_temp
            else:
                ids+=ids_temp
    predictions=numpy.column_stack((ids,predictions))
    _ouput_path=g_config['ouput_path']
    logging.info(f'writing model outputs to \'{_ouput_path}\'')
    write_csv(predictions,_ouput_path)
    _tend=time.time()
    logging.info(f'***** Running wall time: {_tend-_tstart:.2f} s *****')
    logging.info(f'***** {__name__} finished at {_ctime(_tend)} *****')

if __name__=='__main__':
    main()
