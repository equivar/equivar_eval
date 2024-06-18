import os
import sys
import json
import csv
import ase
import ase.io
import logging
import numpy
import torch
import e3nn

# e3nn version 0.4.4 and lower compatibility
_e3nn_ver=[int(_) for _ in e3nn.__version__.split('.')]
if _e3nn_ver[0]==0 and _e3nn_ver[1]<=4:
    import e3nn.o3
    import e3nn.nn
    import e3nn.io

from torch_geometric.data import (
        Data,
        InMemoryDataset,
        )
from equivar_eval.basis import (
        GaussianBasisProjection,
        GaussianCosEnvelopeBasisProjection,
        )

oj=os.path.join

_rng=numpy.random.default_rng(seed=555)

def _cell_vol(cell)->float:
    V=abs(numpy.dot(cell[0],numpy.cross(cell[1],cell[2])))
    return V

class AtomsToGraphs:
    def __init__(
            self,
            path_in,
            graph_max_radius,
            num_radial,
            edge_sh_lmax: int=2,
            radial_basis: str=None,
            ):
        if radial_basis is None:
            radial_basis='Gaussian'
        assert radial_basis in ['Gaussian','GaussianCosEnvelope',]
        self.radial_basis=radial_basis
        logging.info(f'Radial basis: {self.radial_basis}')
        self.path_in=path_in
        if self.path_in is not None:
            assert os.path.isdir(self.path_in)
        self.graph_max_radius=graph_max_radius
        self.num_radial=num_radial
        self.edge_sh_lmax=edge_sh_lmax

    def extxyz_input_iterator(self):
        frames=ase.io.read(oj(self.path_in,'frames.xyz'),index=':',format='extxyz')
        for atoms in frames:
            yield atoms.info['id'],atoms

    def convert(self):
        datas=[]
        if self.radial_basis=='Gaussian':
            distance_expansion=GaussianBasisProjection(
                0.0,
                self.graph_max_radius,
                self.num_radial
                )
        elif self.radial_basis=='GaussianCosEnvelope':
            distance_expansion=GaussianCosEnvelopeBasisProjection(
                0.0,
                self.graph_max_radius,
                self.num_radial
                )
        else:
            raise TypeError
        _irreps=e3nn.o3.Irreps.spherical_harmonics(self.edge_sh_lmax)
        sh=e3nn.o3.SphericalHarmonics(
            _irreps,
            normalize=True
        )
        it=self.extxyz_input_iterator()
        for structure_id,atoms in it:
            data=Data()
            distance_vec=atoms.get_all_distances(mic=True,vector=True)
            r=numpy.linalg.norm(distance_vec,axis=2)
            idx=numpy.where((r<self.graph_max_radius) & (r>0.1))
            edge_attr_radial=distance_expansion(torch.tensor(r[idx],dtype=torch.float))
            edge_attr_angular=sh(torch.tensor(distance_vec[idx]/r[idx].reshape(-1,1),dtype=torch.float))
            _combined_edge_attr=torch.einsum('bi,bj->bij',(edge_attr_radial,edge_attr_angular))
            _ts=[]
            for _sl in e3nn.o3.Irreps.spherical_harmonics(self.edge_sh_lmax).slices():
                _ts.append(torch.flatten(_combined_edge_attr[:,:,_sl],start_dim=-2))
            data.edge_attr=torch.cat(_ts,dim=-1)
            nodes_i=idx[0]
            nodes_j=idx[1]
            data.edge_index=torch.LongTensor(numpy.array([nodes_j,nodes_i],dtype=int))
            data.structure_id=torch.LongTensor(numpy.array([structure_id]*len(atoms),dtype=int))
            data.Z=torch.tensor(atoms.get_atomic_numbers(),dtype=torch.long)
            data.vol=torch.tensor(_cell_vol(atoms.get_cell()))
            data.num_nodes=torch.tensor(len(atoms),dtype=torch.int)
            datas.append(data)
        logging.info(f'data size: {len(datas)}')
        data,slices=InMemoryDataset.collate(datas)
        return data,slices

class InMemoryDatasetUtil(InMemoryDataset):
    def __init__(self,data,slices):
        super().__init__()
        self.data,self.slices=data,slices
