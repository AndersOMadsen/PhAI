import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
import pandas as pd
import numpy as np
import math
import os
import sys, getopt
import random
import crystallography_module

class ConvolutionalBlock(nn.Module):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()

        self.act = nn.GELU()

        self.conv1 = nn.Conv3d(filters, filters, kernel_size = kernel_size, padding = padding)
        self.conv2 = nn.Conv3d(filters, filters, kernel_size = kernel_size, padding = padding)

        self.norm1 = nn.GroupNorm(filters, filters)
        self.norm2 = nn.GroupNorm(filters, filters)

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.act(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.norm2(x)

        x = x + identity
        return x

class MLPLayer(nn.Module):
    def __init__(self, token_nr, dim, dim_exp, mix_type):
        super().__init__()

        self.act    = nn.GELU()

        self.norm1  = nn.GroupNorm(token_nr, token_nr)

        if mix_type == 'token':
            self.layer1 = nn.Conv1d(kernel_size = 1, in_channels = token_nr, out_channels = dim_exp)
            self.layer2 = nn.Conv1d(kernel_size = 1, in_channels = dim_exp, out_channels = token_nr)
        else:
            self.layer1 =  nn.Linear(dim , dim_exp)
            self.layer2 =  nn.Linear(dim_exp, dim)

        self.mix_type = mix_type

    def forward(self, x):
        identity = x

        x = self.norm1(x)

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)

        x = x + identity

        return x

class PhAINeuralNetwork(nn.Module):
    def __init__(self, *, max_index, filters, kernel_size, cnn_depth, dim, dim_exp, dim_token_exp, mlp_depth, reflections):
        super().__init__()

        hkl           = [max_index*2+1, max_index+1, max_index+1]
        mlp_token_nr  = filters
        padding       = int((kernel_size - 1) / 2)

        self.net_a = nn.Sequential(
            Rearrange('b x y z  -> b 1 x y z '),

            nn.Conv3d(1, filters, kernel_size = kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters)
        )

        self.net_p = nn.Sequential(
            Rearrange('b x y z  -> b 1 x y z '),

            nn.Conv3d(1, filters, kernel_size = kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters)
        )

        self.net_convolution_layers = nn.Sequential(
            *[nn.Sequential(
                ConvolutionalBlock(filters, kernel_size = kernel_size, padding = padding),
            ) for _ in range(cnn_depth)],
        )

        self.net_projection_layer = nn.Sequential(
            Rearrange('b c x y z  -> b c (x y z)'),
            nn.Linear(hkl[0]*hkl[1]*hkl[2], dim),
        )

        self.net_mixer_layers = nn.Sequential(
            *[nn.Sequential(
                MLPLayer(mlp_token_nr, dim, dim_token_exp, 'token'),
                MLPLayer(mlp_token_nr, dim, dim_exp      , 'channel'),
            ) for _ in range(mlp_depth)],
            nn.LayerNorm(dim),
        )

        self.net_output = nn.Sequential(
            Rearrange('b t x -> b x t'),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b x 1 -> b x'),

            nn.Linear(dim, reflections*2),
            Rearrange('b (c h) -> b c h ', h = reflections),
        )

    def forward(self, input_amplitudes, input_phases):

        a = self.net_a(input_amplitudes)
        p = self.net_p(input_phases)

        x = a + p

        x = self.net_convolution_layers(x)

        x = self.net_projection_layer(x)

        x = self.net_mixer_layers(x)

        phases = self.net_output(x)

        return phases

def randomize_output(output):
    for j in range(21):
        for jj in range(11):
            for jjj in range(11):
                if random.randint(0, 1) == 1:
                    output[0][j][jj][jjj] = -180.
                else:
                    output[0][j][jj][jjj] = 0.
    return output

def phases(output_phases):
    bin_size = 180.0
    offset   = bin_size / 2
    bin_nr   = int(360 / bin_size)
    output_phases = output_phases.permute(0,2,1)
    output_phases = torch.argmax(output_phases, dim=2)
    return offset + (output_phases*bin_size) - 180.00 - (bin_size/2)


# from shelxl manual. This .hkl should be readable by Olex2:
# m=3: List h,k,l, Fo, σ(Fo), A(real) and B(imag) in Fortran FORMAT(3I4,4F8.2), the reflections
# being processed exactly as for m=2.



fcf_header1 = """
data_PhAI
_shelx_title 'PhAI phased structure factors'
_shelx_refln_list_code          3

loop_
 _space_group_symop_operation_xyz
 'x, y, z'
 '-x, y+1/2, -z+1/2'
 '-x, -y, -z'
 'x, -y-1/2, z-1/2'

"""


fcf_header2 = """
loop_
 _refln_index_h
 _refln_index_k
 _refln_index_l
 _refln_F_meas
 _refln_F_sigma
 _refln_A_calc
 _refln_B_calc

"""


def output_files(amplitudes_ord, output, fname, fname_ext, cellparam):
    #output
    fcf_fname = fname[:-2]+'.fcf'
    file_out = open(fname, 'w')
    fcf_file_out = open(fcf_fname, 'w')
    file_out_ext = open(fname_ext, 'w')

    fcf_file_out.write(fcf_header1)
    fcf_file_out.write('_cell_length_a  %8.3f \n'%cellparam[0])
    fcf_file_out.write('_cell_length_b  %8.3f \n'%cellparam[1])
    fcf_file_out.write('_cell_length_c  %8.3f \n'%cellparam[2])
    fcf_file_out.write('_cell_angle_alpha  %8.3f \n'%cellparam[3])
    fcf_file_out.write('_cell_angle_beta   %8.3f \n'%cellparam[4])
    fcf_file_out.write('_cell_angle_gamma  %8.3f \n'%cellparam[5])
    fcf_file_out.write(fcf_header2)

    for n in range(0, len(hkl_array)):
        #remove locus artefact
        if hkl_array[n][2] == 0 and hkl_array[n][0] < 0:
            continue
        #write
        if amplitudes_ord[n] != 0.:
            if output[0][n] == 0:
                F = complex(amplitudes_ord[n], 0)
            elif output[0][n] == -180:
                F = complex(-amplitudes_ord[n], 0)
            else:
                print('Wrong phase!?', output[0][n])
                input()
            file_out.write('{} {} {} {}\n'.format(*hkl_array[n], F))
            fcf_file_out.write('%4i%4i%4i%8.2f%8.2f%8.2f%8.2f\n'%(hkl_array[n][0],hkl_array[n][1],hkl_array[n][2], abs(F), 0, F.real, 0))
        else:
            #write extended phases
            if output[0][n] == 0:
                F = complex(amplitudes_ord[n], 0)
            elif output[0][n] == -180:
                F = complex(-amplitudes_ord[n], 0)
            else:
                print('Wrong phase!?', output[0][n])
                input()
            file_out_ext.write('{} {} {} {}\n'.format(*hkl_array[n], F))
    fcf_file_out.close()
    file_out.close()    
    file_out_ext.close() 



# model definition
model_args = {
     'max_index' : 10,
       'filters' : 96,
   'kernel_size' : 3,
     'cnn_depth' : 6,
           'dim' : 1024,
       'dim_exp' : 2048,
 'dim_token_exp' : 512,
     'mlp_depth' : 8,
   'reflections' : 1205,
}


model = PhAINeuralNetwork(**model_args)
state = torch.load('./PhAI_model.pth')#, weights_only = True)
model.load_state_dict(state)

max_index = 10
hkl_array = []
for h in range(-max_index, max_index+1):
    for k in range(0, max_index+1):
        for l in range(0, max_index+1):
            if not(h==0 and k==0 and l==0):
                if math.sqrt(h**2+k**2+l**2) <= max_index:
                    hkl_array.append([h,k,l])
hkl_array = np.array(hkl_array,dtype=np.int32)


#print()
#output_files(amplitudes_ord, output, infile[:len(infile)-4] + '.F', infile[:len(infile)-4] + '_phase_extension.F') 






