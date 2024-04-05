import numpy as np
import math
import matplotlib.pyplot as plt
import xraydb
import sys
import time
import fortranformat as ff
import pandas as pd
import os

def is_centrosymm(SG_symm):
    for i in range(len(SG_symm[0])):
        if (SG_symm[0][i]==np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])).all() and np.sum(SG_symm[1][i])==0:
            return True
    return False

def is_centered(SG_symm):
    #is lattice centered?
    for i in range(len(SG_symm[0])):
        if (SG_symm[0][i]==np.identity(3)).all() and np.sum(SG_symm[1][i])!=0:
            return True
    return False        

def get_centering_vectors(SG_symm):
    #get lattice centering vectors
    centering = []
    for i in range(len(SG_symm[0])):
        if (SG_symm[0][i]==np.identity(3)).all() and np.sum(SG_symm[1][i])!=0:
            centering.append(SG_symm[1][i])
    return centering      

def volume(cellparam):
    a = cellparam[0]
    b = cellparam[1]
    c = cellparam[2]
    al = cellparam[3]*(math.pi/180)
    be = cellparam[4]*(math.pi/180)
    ga = cellparam[5]*(math.pi/180)
    return a*b*c*math.sqrt(1-(math.cos(al))**2-(math.cos(be))**2-(math.cos(ga))**2+2*(math.cos(al))*(math.cos(be))*(math.cos(ga)))

def calc_density_map_full(H, F, cellparam, resolution, pixel_mult, N, sort_reflns):
    if sort_reflns == True:
        H, F = sort_reflections(H, F)
    
    #The map is calculated using Beevers-Lipson factorization.
    A = F.real   
    
    #Beevers-Lipson factorization STEP1
    S_array = np.array([[(H[0][0], H[0][1]), np.array([[A[0]]]), np.array([[H[0][2]]])]], dtype=object)
    for m in range(1, len(H)):
        hh = H[m][0]
        kk = H[m][1]
        ll = H[m][2]
        is_in_the_list = False
        for S in S_array:  
            if S[0] == (hh, kk):
                S[1] = np.append(S[1], A[m])
                S[2] = np.append(S[2], ll)
                is_in_the_list = True
        if is_in_the_list == False:
            S_array = np.append(S_array, np.array([[(hh, kk), np.array([[A[m]]]), np.array([[ll]])]], dtype=object), axis=0)
    #S_array = [[(h, k), [A1, A2..], [l1, l2..]], [..]]

    #END Beevers-Lipson factorization STEP1
    
    na = int(round(cellparam[0]/resolution))
    nb = int(round(cellparam[1]/resolution))
    nc = int(round(cellparam[2]/resolution))

    if na % pixel_mult[0] != 0:
        na = na + pixel_mult[0] - (na % pixel_mult[0])
    if nb % pixel_mult[1] != 0:
        nb = nb + pixel_mult[1] - (nb % pixel_mult[1])
    if nc % pixel_mult[2] != 0:
        nc = nc + pixel_mult[2] - (nc % pixel_mult[2])
        
    vol = volume(cellparam)
    
    den_map = np.zeros((na, nb, nc), dtype=float)
    

    #sys.stdout.write('\n')
    pi = math.pi 
    for nc_step in range(nc):
        proc = str(round((100*nc_step/(nc-1)))) + '%'
        sys.stdout.write('calculating density map [res = '+ str(resolution) +'Å]: ' + '%s\r' % proc)
        z = (1/nc)*nc_step
        # make the map symmetrical with respect to the voxel grid:
        # if nc_step / nc >= 0.5:
            # z = nc_step / nc
        # else:
            # z = (nc_step+1)/nc
            
        #Beevers-Lipson factorization STEP2
        
        #S_array = [[(h, k), [A1, A2..], [l1, l2..]], [..]]  
        h_ind_reduced = [S_array[0][0][0]]
        k_ind_reduced = [S_array[0][0][1]]
        S1 = [np.sum(S_array[0][1]*np.cos(2*pi*S_array[0][2]*z))]
        S2 = [np.sum(-S_array[0][1]*np.sin(2*pi*S_array[0][2]*z))]
        S3 = [np.sum(-S_array[0][1]*np.sin(2*pi*S_array[0][2]*z))]
        S4 = [np.sum(-S_array[0][1]*np.cos(2*pi*S_array[0][2]*z))]        
        for m in range(1, len(S_array)):
            h_ind_reduced.append(S_array[m][0][0])
            k_ind_reduced.append(S_array[m][0][1])
            S1.append(np.sum(S_array[m][1]*np.cos(2*pi*S_array[m][2]*z)))
            S2.append(np.sum(-S_array[m][1]*np.sin(2*pi*S_array[m][2]*z)))
            S3.append(np.sum(-S_array[m][1]*np.sin(2*pi*S_array[m][2]*z)))
            S4.append(np.sum(-S_array[m][1]*np.cos(2*pi*S_array[m][2]*z)))
        
        
        #T_array = np.array([[h_ind_reduced[0], np.array([[S1[m]]]), np.array([[S2[m]]]), np.array([[S3[m]]]), np.array([[S4[m]]]), np.array([[k_ind_reduced[0]]])]], dtype=object)
        T_array = np.array([[h_ind_reduced[0], np.array([[S1[0]]]), np.array([[S2[0]]]), np.array([[S3[0]]]), np.array([[S4[0]]]), np.array([[k_ind_reduced[0]]])]], dtype=object)
        for m in range(1, len(h_ind_reduced)):
            is_in_the_list = False
            for mm in range(len(T_array)):
                if T_array[mm][0] == h_ind_reduced[m]:
                    T_array[mm][1] = np.append(T_array[mm][1], S1[m])
                    T_array[mm][2] = np.append(T_array[mm][2], S2[m])
                    T_array[mm][3] = np.append(T_array[mm][3], S3[m])
                    T_array[mm][4] = np.append(T_array[mm][4], S4[m])
                    T_array[mm][5] = np.append(T_array[mm][5], k_ind_reduced[m])
                    is_in_the_list = True
            if is_in_the_list == False:
                T_array = np.append(T_array, np.array([[h_ind_reduced[m], np.array([[S1[m]]]), np.array([[S2[m]]]), np.array([[S3[m]]]), np.array([[S4[m]]]), np.array([[k_ind_reduced[m]]])]], dtype=object), axis=0)
                    
        #T_array = [[h, [S1_1, S1_2..], [S2_1, S2_2..], [S3_1, S3_2..], [S4_1, S4_2..], [k1, k2..]], [..]] 
        
        #END Beevers-Lipson factorization STEP2

        for nb_step in range(nb):
        
            y = (1/nb)*nb_step
            # make the map symmetrical with respect to the voxel grid:
            # if nb_step / nb >= 0.5:
                # y = nb_step / nb
            # else:
                # y = (nb_step+1)/nb
            
            #Beevers-Lipson factorization STEP3
            #T_array = [[h, [S1_1, S1_2..], [S2_1, S2_2..], [S3_1, S3_2..], [S4_1, S4_2..], [k1, k2..]], [..]] 
            h_ind_reduced_reduced = [T_array[0][0]]
            T1 = [np.sum(np.cos(2*pi*T_array[0][5]*y)*T_array[0][1]+np.sin(2*pi*T_array[0][5]*y)*T_array[0][2])]
            T2 = [np.sum(np.cos(2*pi*T_array[0][5]*y)*T_array[0][3]+np.sin(2*pi*T_array[0][5]*y)*T_array[0][4])]
            for mm in range(1, len(T_array)):
                h_ind_reduced_reduced.append(T_array[mm][0])
                T1.append(np.sum(np.cos(2*pi*T_array[mm][5]*y)*T_array[mm][1]+np.sin(2*pi*T_array[mm][5]*y)*T_array[mm][2]))
                T2.append(np.sum(np.cos(2*pi*T_array[mm][5]*y)*T_array[mm][3]+np.sin(2*pi*T_array[mm][5]*y)*T_array[mm][4]))
            
            #END Beevers-Lipson factorization STEP3
            
            for na_step in range(na):
                x = (1/na)*na_step
                # make the map symmetrical with respect to the voxel grid:
                # if na_step / na >= 0.5:
                    # x = na_step / na
                # else:
                    # x = (na_step+1)/na
                
                #Beevers-Lipson factorization STEP4
                h_ind_reduced_reduced = np.asarray(h_ind_reduced_reduced)
                T1 = np.asarray(T1)
                T2 = np.asarray(T2)
                rho = (N/vol)*(np.sum(np.cos(2*pi*h_ind_reduced_reduced*x)*T1+np.sin(2*pi*h_ind_reduced_reduced*x)*T2))
                #END Beevers-Lipson factorization STEP4

                
                den_map[na_step][nb_step][nc_step] = rho

    return den_map   


def save_den_map_xplor(den_map, cellparam, outputfile, name):
    
    final_string = []
    #xplor header
    na = den_map.shape[0]
    nb = den_map.shape[1]
    nc = den_map.shape[2]
    final_string.append('\n       1\n' + name + '\n')
    line_out = ff.FortranRecordWriter('(9I8)')
    final_string.append(line_out.write([na, 0, na-1, nb, 0, nb-1, nc, 0, nc-1]) + '\n')
    line_out = ff.FortranRecordWriter('(6E12.5)')
    final_string.append(line_out.write([cellparam[0], cellparam[1], cellparam[2], cellparam[3], cellparam[4], cellparam[5]]) + '\n')
    final_string.append('ZYX') 
    
    sys.stdout.write('\n')
    for nc_step in range(nc):
        proc = str(round((100*nc_step/(nc-1)))) + '%'
        sys.stdout.write('writing density map: ' + '%s\r' % proc)
        if nc_step == 0:
            final_string.append('\n')
        else:
            if ((na * nb) % 6) != 0:
                final_string.append('\n')
        line_out = ff.FortranRecordWriter('(I8)')
        final_string.append(line_out.write([nc_step]) + '\n')
        already6 = 0
        for nb_step in range(nb):
            for na_step in range(na):
                line_out = ff.FortranRecordWriter('(E12.5)')
                final_string.append(line_out.write([den_map[na_step][nb_step][nc_step]]))
                already6 = already6 + 1
                if already6 == 6:
                    already6 = 0
                    final_string.append('\n')

    final_string.append('\n')
    line_out = ff.FortranRecordWriter('(I8)')
    final_string.append(line_out.write([-9999]) + '\n')
    line_out = ff.FortranRecordWriter('(2(E12.4,1X))')
    rho_mean = np.average(den_map)
    rho_std = 0. #rho_std = statistics.stdev(den_map)
    final_string.append(line_out.write([rho_mean, rho_std]))
    
    file_out = open(outputfile, 'w')
    file_out.write(''.join(final_string))
    file_out.close()


def calculate_F(H, cellparam, atoms, SG_symm, rel_scale, F_format, outputfile, do_return):
    #calculate exact F

    #omit inversion-induced operators
    symmR = SG_symm[0]
    symmT = SG_symm[1]
    mult = 1
    centrosymm = is_centrosymm(SG_symm)
    if centrosymm == True:
        uniqueR = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        uniqueT = np.array([[0, 0, 0]])
        for i in range(len(symmR)):
            already_listed = False
            for j in range(len(uniqueR)):
                if (symmR[i]==uniqueR[j]).all() and (symmT[i]==uniqueT[j]).all():
                    already_listed = True
                    continue
                if (np.matmul(symmR[i], np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))==uniqueR[j]).all() and (symmT[i]==uniqueT[j]).all():
                    already_listed = True
                    break
            if already_listed == False:
                uniqueR = np.append(uniqueR, np.array([symmR[i]]), axis=0)
                uniqueT = np.append(uniqueT, np.array([symmT[i]]), axis=0)
        mult = mult * (len(symmR) / len(uniqueR))
        symmR = np.copy(uniqueR)
        symmT = np.copy(uniqueT)

    #omit centering-induced operators

    #prepare H-independent matrices
    ai = np.array([])
    X = np.array([[0, 0, 0]])
    atom_type = np.array([])
    U = np.array([])
    for i in range(len(atoms)):
        ai = np.append(ai, atoms[i].ai)
        X = np.append(X, np.array([atoms[i].X]), axis=0)
        atom_type = np.append(atom_type, atoms[i].symbol)
        U = np.append(U, atoms[i].U)
    X = np.delete(X, 0, axis=0)
    
    #prepare H-dependent matrices: d/q
    xdb = xraydb.get_xraydb()
    q = np.array([]) #=Q/4pi
    d = np.array([])
    for i in range(len(H)):
        dsp = d_spacing(H[i], cellparam)
        q = np.append(q, 1 / (2 * dsp))
        d = np.append(d, dsp)    
    
    #prepare H-dependent matrices: set of atom types
    for atom in atom_type:
        if atom == 'D': #is D form factor in the tables?
            atom = 'H'
    types = []
    for at in atom_type:
        is_in_types = False
        for at2 in types:
            if at == at2:
                is_in_types = True
                break
        if is_in_types == False:
            types.append(str(at))
     
    #prepare H-dependent matrices: f0 array
    f0arr = []
    for at in types:
        f0arr.append(xdb.f0(str(at), q))

    #calculate_F
    F = np.array([])
    for i in range(len(H)):    
        prc = str(round(100*(i+1)/len(H))) + '%'
        sys.stdout.write('generating Fs: ' + '%s\r' % prc)        
        farr = np.array([])
        for atom in atom_type:
            farr = np.append(farr, f0arr[types.index(str(atom))][i])
        temp_f = np.exp(-8*((math.pi)**2)*U*(1/(2*d[i]))**2) #np.exp(-atom_B*(1/(2*d[i]))**2)
        
        A = 0
        B = 0
        for j in range(len(atom_type)):
            for s in range(len(symmR)):
                A = A + ai[j] * farr[j] * temp_f[j] * np.cos(2 * math.pi * (np.matmul(np.matmul(H[i], symmR[s]), X[j]) + np.matmul(H[i], symmT[s])))
                if centrosymm == False:
                    B = B + ai[j] * farr[j] * temp_f[j] * np.sin(2 * math.pi * (np.matmul(np.matmul(H[i], symmR[s]), X[j]) + np.matmul(H[i], symmT[s])))
        
        F = np.append(F, complex(mult*A, mult*B))
        #print(F)
        
    
    #scale
    if rel_scale == True:
        if centrosymm == True:
            coeff = max(np.abs(F))
            for i in range(len(F)):
                F[i] = complex(F[i].real / coeff, F[i].imag)
        else:
            print('Scaling for non-centrosymmetric structures: Fix this!!!')
            input()
    
    #output
    if outputfile != None:
        if F_format == 1:
            fout = open(outputfile, 'w')
            for i in range(len(F)):
                fout.write('{} {} {} {}\n'.format(H[i][0], H[i][1], H[i][2], F[i]))
            fout.close()
    
    if do_return == True:
        return F

def gen_hkl_ovoid(cellparam, max_sin_th_over_lambda, system, omit_absences, SG_symm):
    #generate list of hkl based on desired resolution
    maxdim = max(cellparam[0:2])
    min_d = (1 / max_sin_th_over_lambda) / 2
    radius = int((maxdim/min_d)*2) #radius estimation
    H = np.array([[0, 0, 0]], dtype=int)
    
    if system == 'triclinic':
        for hh in range(-radius, radius+1):
            for kk in range(-radius, radius+1):
                for ll in range(0, radius+1):
                    if not(hh==0 and kk==0 and ll==0):
                        if math.sqrt(hh**2+kk**2+ll**2) <= radius: #sphere
                            #if not(hh < 0 and kk < 0 and ll==0): #locus
                            if not(kk > 0 and ll==0): #locus
                                if not(hh < 0 and kk == 0 and ll==0): #locus
                                    if 1 / (2*d_spacing([hh,kk,ll], cellparam)) <= max_sin_th_over_lambda:
                                        if omit_absences == True:
                                            if is_absent([hh, kk, ll], SG_symm) == False:
                                                H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
                                        else:
                                            H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
        H = np.delete(H, 0, axis=0)
        #plot_hkl_ovoid(H)

    if system == 'monoclinic':
        for hh in range(-radius, radius+1):
            for kk in range(0, radius+1):
                for ll in range(0, radius+1):
                    if not(hh==0 and kk==0 and ll==0):
                        if math.sqrt(hh**2+kk**2+ll**2) <= radius: #sphere
                            if not(hh < 0 and ll==0): #locus
                                if 1 / (2*d_spacing([hh,kk,ll], cellparam)) <= max_sin_th_over_lambda:
                                    if omit_absences == True:
                                        if is_absent([hh, kk, ll], SG_symm) == False:
                                            H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
                                    else:
                                        H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
        H = np.delete(H, 0, axis=0)
    
    return H
    

def gen_hkl_shell(cellparam, min_sin_th_over_lambda, max_sin_th_over_lambda, system, omit_absences, SG_symm):
    #generate list of hkl based on desired resolution
    maxdim = max(cellparam[0:2])
    min_d = (1 / max_sin_th_over_lambda) / 2
    radius = int((maxdim/min_d)*2) #radius estimation
    H = np.array([[0, 0, 0]], dtype=int)
    
    if system == 'triclinic':
        for hh in range(-radius, radius+1):
            for kk in range(-radius, radius+1):
                for ll in range(0, radius+1):
                    if not(hh==0 and kk==0 and ll==0):
                        if math.sqrt(hh**2+kk**2+ll**2) <= radius: #sphere
                            if not(hh < 0 and kk < 0 and ll==0): #locus
                                if 1 / (2*d_spacing([hh,kk,ll], cellparam)) <= max_sin_th_over_lambda and 1 / (2*d_spacing([hh,kk,ll],cellparam)) > min_sin_th_over_lambda:
                                    if omit_absences == True:
                                        if is_absent([hh, kk, ll], SG_symm) == False:
                                            H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
                                    else:
                                        H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
        H = np.delete(H, 0, axis=0)

    if system == 'monoclinic':
        for hh in range(-radius, radius+1):
            for kk in range(0, radius+1):
                for ll in range(0, radius+1):
                    if not(hh==0 and kk==0 and ll==0):
                        if math.sqrt(hh**2+kk**2+ll**2) <= radius: #sphere
                            if not(hh < 0 and ll==0): #locus
                                if 1 / (2*d_spacing([hh,kk,ll], cellparam)) <= max_sin_th_over_lambda and 1 / (2*d_spacing([hh,kk,ll],cellparam)) > min_sin_th_over_lambda:
                                    if omit_absences == True:
                                        if is_absent([hh, kk, ll], SG_symm) == False:
                                            H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
                                    else:
                                        H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
        H = np.delete(H, 0, axis=0)
    
    return H

def deduce_completeness(H_present, shell_increment, cellparam, system, omit_absences, SG_symm):
    completeness = []
    shell_limits = [(0, shell_increment)]
    H_present = H_present.tolist()
    while len(H_present) != 0:
        i = len(shell_limits) - 1
        H_in_shell = gen_hkl_shell(cellparam, shell_limits[i][0], shell_limits[i][1], system=system, omit_absences=omit_absences, SG_symm=SG_symm)
        #print(shell_limits[i])
        #plot_hkl_ovoid(H_in_shell)
        present_in_shell = 0
        for j in range(len(H_in_shell)):
            for k in range(len(H_present)):
                if (H_in_shell[j] == H_present[k]).all():
                    present_in_shell += 1
                    del H_present[k]
                    break
        if len(H_in_shell) == 0:
            completeness.append(1.)
        else:
            completeness.append(present_in_shell / len(H_in_shell))
        if len(H_present) != 0:
            shell_limits.append((shell_limits[i][1], shell_limits[i][1] + shell_increment))
    
    #print(shell_limits)
    #print(completeness)
    return shell_limits, completeness
    
def plot_hkl_ovoid(H):
    #plot a list of hkl
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([h[0] for h in H], [h[1] for h in H], [h[2] for h in H], marker='.')
    ax.set_xlabel('$h$')
    ax.set_ylabel('$k$')
    ax.set_zlabel('$l$')
    plt.show()                

def get_laue(SG_symm):
#FIX!!!!!!!!!!!!!!!!
    if len(SG_symm[0]) == 2:
        return '-1'
    else:
        return '2/m'

def sort_reflections(H, F):
    ind_H = np.lexsort((H[:, 0], H[:, 1], H[:, 2]))
    H_sorted = np.array([[0, 0, 0]])
    F_sorted = np.array([])
    for i in ind_H:
        H_sorted = np.append(H_sorted, np.array([H[i]]), axis=0)
        F_sorted = np.append(F_sorted, F[i])
    H_sorted = np.delete(H_sorted, 0, axis=0)
    
    return H_sorted, F_sorted

def complete_hkl(H, F, SG_symm, half):
    #symmetrize hkl and F list
    
    #deduce reduced generators, i.e. omit centering- and inversion-induced operators
    centered_lattice = is_centered(SG_symm)
    centrosymm = is_centrosymm(SG_symm)
    if centered_lattice == True:
        centering = get_centering_vectors(SG_symm)
    
    #omit centering-induced operators
    uniqueR = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    uniqueT = np.array([[0, 0, 0]])
    if centered_lattice == True:
        for i in range(len(SG_symm[0])):
            if (SG_symm[0][i]==np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all() and (SG_symm[1][i]==np.array([[0, 0, 0]])).all():
                continue
            already_listed = False
            for j in range(len(uniqueR)):
                if (SG_symm[0][i]==uniqueR[j]).all() and (SG_symm[1][i]==uniqueT[j]).all() == False:
                    for k in range(len(centering)):
                        diff = SG_symm[1][i] - centering[k]
                        diff[0] = modulo([diff[0]], 1)
                        diff[1] = modulo([diff[1]], 1)
                        diff[2] = modulo([diff[2]], 1)
                        if (diff==uniqueT[j]).all():
                            already_listed = True
                            break
                    if already_listed == True:
                        break
            if already_listed == False:
                uniqueR = np.append(uniqueR, np.array([SG_symm[0][i]]), axis=0)
                uniqueT = np.append(uniqueT, np.array([SG_symm[1][i]]), axis=0)
        symmR = np.copy(uniqueR)
        symmT = np.copy(uniqueT)
    else:
        symmR = SG_symm[0]
        symmT = SG_symm[1]

    #omit inversion-induced operators
    uniqueR = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    uniqueT = np.array([[0, 0, 0]])
    if centrosymm == True:
        for i in range(len(symmR)):
            already_listed = False
            for j in range(len(uniqueR)):
                if (symmR[i]==uniqueR[j]).all() and (symmT[i]==uniqueT[j]).all():
                    already_listed = True
                    continue
                if (np.matmul(symmR[i], np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))==uniqueR[j]).all():
                    if (symmT[i]==uniqueT[j]).all():
                        already_listed = True
                        continue
                    if centered_lattice == True:
                        if (symmT[i]==uniqueT[j]).all():
                            already_listed = True
                            continue
                        for k in range(len(centering)):
                            transl = centering[k] + symmT[i]
                            transl[0] = modulo([transl[0]], 1)
                            transl[1] = modulo([transl[1]], 1)
                            transl[2] = modulo([transl[2]], 1)
                            if (transl==uniqueT[j]).all():
                                already_listed = True
                                break
            if already_listed == False:
                uniqueR = np.append(uniqueR, np.array([symmR[i]]), axis=0)
                uniqueT = np.append(uniqueT, np.array([symmT[i]]), axis=0)
        symmR = np.copy(uniqueR)
        symmT = np.copy(uniqueT)    
    
    F_completed = np.copy(F)
    H_completed = np.copy(H)
    
    laue = get_laue(SG_symm)
    

    if laue == '-1':
        for i in range(len(H)):
            #apply Friedel's law
            h_new = np.matmul(np.array(-1*np.identity(3), dtype=int), H[i])
            phi = np.angle(F[i])
            ampl = np.abs(F[i])
            F_completed = np.append(F_completed, ampl * (math.cos(-phi) + math.sin(-phi)*1j))
            H_completed = np.append(H_completed, np.array([h_new]), axis=0)
            
    
    if laue == '2/m':
        locus = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        symmR = np.delete(symmR, 0, axis=0)
        symmT = np.delete(symmT, 0, axis=0)
        
        for i in range(len(H)):
            #apply Friedel's law
            h_new = np.matmul(np.array(-1*np.identity(3), dtype=int), H[i])
            phi = np.angle(F[i])
            ampl = np.abs(F[i])
            F_completed = np.append(F_completed, ampl * (math.cos(-phi) + math.sin(-phi)*1j))
            H_completed = np.append(H_completed, np.array([h_new]), axis=0)
            
            symmetrize = True #assume further symmetrization
            if 0 in H[i]: #possible locus
                for j in range(len(locus)):
                    h = np.array([0, 0, 0])
                    if H[i][0] != 0:
                        h[0] = 1
                    if H[i][1] != 0:
                        h[1] = 1
                    if H[i][2] != 0:
                        h[2] = 1
                    if (h==locus[j]).all():
                        symmetrize = False #locus, do not symmetrize
                        break
            if symmetrize == True:
                for s in range(len(symmR)):
                    h_new = np.matmul(symmR[s], H[i])
                    phi = np.angle(F[i])
                    ampl = np.abs(F[i])
                    phi_new = modulo([phi - 2*math.pi*np.dot(H[i], symmT[s])], 2*math.pi)
                    F_completed = np.append(F_completed, ampl * (math.cos(phi_new) + math.sin(phi_new)*1j))
                    #F_completed = np.append(F_completed, ampl * cmath.exp(1j*phi_new))
                    H_completed = np.append(H_completed, np.array([h_new]), axis=0)
                    
                    #Friedel's law
                    h_new = np.matmul(np.matmul(symmR[s], np.array(-1*np.identity(3), dtype=int)), H[i])
                    phi_new = modulo([-phi_new], 2*math.pi)
                    F_completed = np.append(F_completed, ampl * (math.cos(phi_new) + math.sin(phi_new)*1j))
                    #F_completed = np.append(F_completed, ampl * cmath.exp(1j*phi_new))
                    H_completed = np.append(H_completed, np.array([h_new]), axis=0)
    
    
    #omit half
    if half == True:
        H_half = np.array([[0, 0, 0]])
        F_half = np.array([])
        for i in range(len(H_completed)):
            if H_completed[i][0] > 0:
                H_half = np.append(H_half, np.array([H_completed[i]]), axis=0)
                F_half = np.append(F_half, F_completed[i])
            elif H_completed[i][0] == 0:
                if H_completed[i][1] > 0:
                    H_half = np.append(H_half, np.array([H_completed[i]]), axis=0)
                    F_half = np.append(F_half, F_completed[i])
                elif H_completed[i][1] == 0:
                    if H_completed[i][2] > 0:
                        H_half = np.append(H_half, np.array([H_completed[i]]), axis=0)
                        F_half = np.append(F_half, F_completed[i])
        H_half = np.delete(H_half, 0, axis=0)
        return H_half.astype('int32'), F_half    
    else:
        return H_completed.astype('int32'), F_completed                
        

def modulo(xarr, mod):
    res = np.array([])
    for x in xarr:
        res = np.append(res, x - mod*(x // mod))
    if len(res) == 1:
        return res[0]
    else:
        return res  
        
def modulo_upper(xarr, mod):
    res = np.array([])
    for x in xarr:
        x_new = x - mod*(x // mod)
        if x_new == 0.:
            x_new = mod
        res = np.append(res, x_new)
    if len(res) == 1:
        return res[0]
    else:
        return res  

def translate_F(F, H, t):
    F_t = np.array([])
    phi = np.angle(F)
    phi_new = translate_phi(phi, H, t)
    for j in range(len(F)):
        F_t = np.append(F_t, np.abs(F[j]) * (math.cos(phi_new[j]) + math.sin(phi_new[j])*1j))
    return F_t

def translate_phi(phi, H, t):
    phi_t = np.array([])
    for j in range(len(phi)):
        phi_t = np.append(phi_t, phi[j] - 2*math.pi * np.dot(H[j], t))
    return modulo(phi_t, 2*math.pi)

def get_symm_site_multiplicity(atomX, symmR, symmT):
    atoms_tmp = [atomX]
    for i in range(len(symmR)):
        id_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        id_transl = np.array([0, 0, 0])
        if not((symmR[i]==id_matrix).all() and (symmT[i]==id_transl).all()):
            xyz = np.add(np.matmul(symmR[i], np.array(atomX)), symmT[i])
            x = modulo([xyz[0]], 1)
            y = modulo([xyz[1]], 1)
            z = modulo([xyz[2]], 1)
            atoms_tmp.append([x, y, z])
    unique_atoms = [atoms_tmp[0]]
    for atom2 in atoms_tmp:
        is_unique = True
        for unique_atom in unique_atoms:
            if atom2 == unique_atom:
                is_unique = False
                break
        if is_unique == True:
            unique_atoms.append(atom2)                
    return len(unique_atoms)

def str_entires_to_SG_symm(symm_op_strings):
    col_idx = -1
    symm_op_entries = []    
    for i in range(len(symm_op_strings)):
        symm_op_strings[i] = symm_op_strings[i].replace(' ', '').replace('\t', '').replace('\'', '').replace('"', '').replace(';', ',').replace('0.5', '1/2').replace('0.25', '1/4').replace('0.75', '3/4').lower()
        if col_idx != -1:
            if symm_op_strings[i][0] == '-':
                if len(str(col_idx)) == 1:
                    symm_op_strings[i] = symm_op_strings[i][2:]
                else:
                    symm_op_strings[i] = input('Symm. string for ['+ symm_op_strings[i] +']?')
            else:    
                symm_op_strings[i] = symm_op_strings[i][len(str(col_idx)):]
            col_idx += 1
        symm_op_entries.append(symm_op_strings[i].split(','))
    symmR = np.zeros(shape=(len(symm_op_entries), 3, 3))
    symmT = np.zeros(shape=(len(symm_op_entries), 3))
    
    
    for i, symm_op_entry in enumerate(symm_op_entries):
        W = np.zeros(shape=(3, 3))
        w = np.zeros(shape=(3))
        for j, dim in enumerate(symm_op_entry):
            Wtmp = np.array([[0, 0, 0]])
            if '-x' in dim:
                Wtmp[0][0] = -1
            elif 'x' in dim:
                Wtmp[0][0] = 1
            if '-y' in dim:
                Wtmp[0][1] = -1
            elif 'y' in dim:
                Wtmp[0][1] = 1
            if '-z' in dim:
                Wtmp[0][2] = -1
            elif 'z' in dim:
                Wtmp[0][2] = 1
            dim = dim.replace('-x', '')
            dim = dim.replace('+x', '')
            dim = dim.replace('-y', '')
            dim = dim.replace('+y', '')
            dim = dim.replace('-z', '')
            dim = dim.replace('+z', '')
            dim = dim.replace('x', '')
            dim = dim.replace('y', '')
            dim = dim.replace('z', '')
            dim = dim.lstrip('+')
            
            if not('/' in dim):
                dim = ''
            
            if len(dim) != 0:
                dim = dim.split('/')
                wtmp = float(dim[0]) / float(dim[1])
            else:
                wtmp = 0.0
            W[j] = Wtmp
            w[j] = wtmp
        symmR[i] = W
        symmT[i] = w

    return symmR, symmT
    
    

def SG_symm_to_str_entries(SG_symm, str_format, separator):
    #separator = ' ' #',' ', ' ';' ...
    if str_format == 0:
        basis = ['x', 'y', 'z']
    elif str_format == 1:
        basis = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    symmR = SG_symm[0]
    symmT = SG_symm[1]
    symm_op_entries = []
    for i in range(len(symmR)):
        symm_op_entry = ''
        for j in range(len(symmR[i])):
            for k in range(len(symmR[i])):
                if symmR[i][j][k]==1:
                    if symm_op_entry=='':
                        symm_op_entry += basis[k]
                    elif symm_op_entry[-1] == separator[-1]:
                        symm_op_entry += basis[k]
                    else:
                        symm_op_entry += '+' + basis[k]
                elif symmR[i][j][k]==-1:
                    symm_op_entry += '-' + basis[k]
            if symmT[i][j] != 0:
                symm_op_entry += '+' + str(symmT[i][j])
            if j != len(symmR[i])-1:
                symm_op_entry += separator
        symm_op_entries.append(symm_op_entry)
    return symm_op_entries
        
def read_F(fname):
    if os.stat(fname).st_size == 0:
        return np.array([]), np.array([])
    df = pd.read_csv(fname, delimiter=' ', header=None)
    H = np.array([[0, 0, 0]])
    F = np.array([])
    for i in range(len(df)):
        H = np.append(H, np.array([[df.loc[i,0].astype(int), df.loc[i,1].astype(int), df.loc[i,2].astype(int)]]), axis=0)
        F = np.append(F, complex(df.loc[i,3]))
    H = np.delete(H, 0, axis=0)   
    return H, F

def phase_error(F1, F2, H, semivariants):
    if len(F1) != len(F2):
        print('Error: Structure factor lists are not the same size! [phase_error]')
        input()
    errors = []
    r = []
    phi1 = modulo(np.angle(F1), 2*math.pi)
    phi2 = modulo(np.angle(F2), 2*math.pi)
    w = np.abs(F1)
    for semiv in semivariants:
        phi_t = translate_phi(phi1, H, semiv)
        num = 0
        denom = 0
        num_r = 0
        denom_r = 0
        for i in range(len(F1)):
            num_r += w[i]**2 * math.cos(phi_t[i] - phi2[i])
            denom_r += w[i]**2
            num += w[i] * abs(phi_t[i] - phi2[i])
            denom += w[i] * math.pi
        errors.append(num / denom)
        r.append(num_r / denom_r)
    best_idx = errors.index(min(errors))
    #print()
    #print(errors)
    #print(semivariants)
    #print(best_idx)
    #best_idx = int(input('Best idx? '))
    #input()
    return errors[best_idx], semivariants[best_idx], r[best_idx]



def return_rlfns_w_incorrect_phases(F1, F2, H, semivariant, cellparam):
    phi1 = modulo(np.angle(F1), 2*math.pi)
    phi2 = modulo(np.angle(F2), 2*math.pi)
    phi_t = translate_phi(phi1, H, semivariant)
    incorrect_H = np.array([[0, 0, 0]])
    incorrect_q = np.array([])
    incorrect_Fabs = np.array([])
    for i in range(len(F1)):
        if phi_t[i] != phi2[i]:
            incorrect_H = np.append(incorrect_H, np.array([[H[i][0], H[i][1], H[i][2]]]), axis=0)
            incorrect_q = np.append(incorrect_q, 1/(2*d_spacing(H[i], cellparam)))
            incorrect_Fabs = np.append(incorrect_Fabs, np.abs(F1[i]))
    incorrect_H = np.delete(incorrect_H, 0, axis=0) 
    
    return incorrect_H, incorrect_q, incorrect_Fabs
    
            
def d_spacing(H, cellparam):
    a = cellparam[0]
    b = cellparam[1]
    c = cellparam[2]
    al = cellparam[3]*(math.pi/180)
    be = cellparam[4]*(math.pi/180)
    ga = cellparam[5]*(math.pi/180)
    h = H[0]
    k = H[1]
    l = H[2]
    
    #Triclinic (general case)
    s11 = b**2 * c**2 * (math.sin(al))**2
    s22 = a**2 * c**2 * (math.sin(be))**2
    s33 = a**2 * b**2 * (math.sin(ga))**2
    s12 = a * b * c**2 * (math.cos(al)*math.cos(be) - math.cos(ga))
    s23 = a**2 * b * c * (math.cos(be)*math.cos(ga) - math.cos(al))
    s31 = a * b**2 * c * (math.cos(ga)*math.cos(al) - math.cos(be))
    vol_sq = a**2 * b**2 * c**2 * (1 - (math.cos(al))**2 - (math.cos(be))**2 - (math.cos(ga))**2 + 2 * (math.cos(al)) * (math.cos(be)) * (math.cos(ga)))
    temp_d = (1/vol_sq) * (s11*h**2 + s22*k**2 + s33*l**2 + 2*s12*h*k + 2*s23*k*l + 2*s31*l*h)
    return math.sqrt(1/temp_d) #returns d_spacing

def cellparam_to_matrix(cellparam):
    a = cellparam[0]
    b = cellparam[1]
    c = cellparam[2]
    al = cellparam[3]*(math.pi/180)
    be = cellparam[4]*(math.pi/180)
    ga = cellparam[5]*(math.pi/180)
    
    v = a*b*c*math.sqrt(1-(math.cos(al))**2-(math.cos(be))**2-(math.cos(ga))**2+2*(math.cos(al))*(math.cos(be))*(math.cos(ga)))
    
    cos_al_st = (math.cos(be)*math.cos(ga)-math.cos(al))/(math.sin(be)*math.sin(ga))
    c_st = (a * b * math.sin(ga)) / v

    lattice = np.array([[a, 0, 0],
           [b * math.cos(ga), b * math.sin(ga), 0],
           [c * math.cos(be), -c * math.sin(be) * cos_al_st, 1/c_st]])
    
    return lattice
    
def matrix_to_cellparam(lattice):
    cell_a = math.sqrt(lattice[0][0]**2 + lattice[0][1]**2 + lattice[0][2]**2)
    cell_b = math.sqrt(lattice[1][0]**2 + lattice[1][1]**2 + lattice[1][2]**2)
    cell_c = math.sqrt(lattice[2][0]**2 + lattice[2][1]**2 + lattice[2][2]**2)
    cell_al = math.acos(np.dot(lattice[1], lattice[2]) / (cell_b * cell_c))*(180/math.pi)
    cell_be = math.acos(np.dot(lattice[0], lattice[2]) / (cell_a * cell_c))*(180/math.pi)
    cell_ga = math.acos(np.dot(lattice[0], lattice[1]) / (cell_a * cell_b))*(180/math.pi)
    
    return [cell_a, cell_b, cell_c, cell_al, cell_be, cell_ga]

def is_absent(H, SG_symm):
    #P21/c #generalize this!!!
    h = H[0]
    k = H[1]
    l = H[2]
    
    if h == 0 and l == 0:
        if (k % 2) != 0:
            return True
    if k == 0:
        if (l % 2) != 0:
            return True
    return False

def allowed_origin_shifts(SG_symm):
    #add R-lattice handling
    P_semivariants = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]
    X_semivariants = []
    if is_centered(SG_symm):
        centering = get_centering_vectors(SG_symm)
        centering.append(np.array([0, 0, 0]))
        for centering_vector in centering:
            for P_semivariant in P_semivariants:
                X_semivariants.append([P_semivariant[0] + centering_vector[0]/2, P_semivariant[1] + centering_vector[1]/2, P_semivariant[2] + centering_vector[2]/2])
        return X_semivariants
    else:
        return P_semivariants

def sys_abs(symmR, symmT): #under construction derive sys absence strings
    ident = np.identity(3)
    cond_str_arr = []
    for i in range(len(symmR)):
        if (ident==symmR[i]).all() and (np.array([0, 0, 0])==symmT[i]).all():
            continue
        currR = np.add(ident, symmR[i]) / 2
        if currR[0][0] == 1:
            cond_str1 = 'h'
        elif currR[0][1] == 1:
            cond_str1 = 'k'
        elif currR[0][2] == 1:
            cond_str1 = 'l'
        else:
            cond_str1 = '0'
        if currR[1][0] == 1:
            cond_str2 = 'h'
        elif currR[1][1] == 1:
            cond_str2 = 'k'
        elif currR[1][2] == 1:
            cond_str2 = 'l'
        else:
            cond_str2 = '0'
        if currR[2][0] == 1:
            cond_str3 = 'h'
        elif currR[2][1] == 1:
            cond_str3 = 'k'
        elif currR[2][2] == 1:
            cond_str3 = 'l'
        else:
            cond_str3 = '0'
        cond_str = cond_str1 + cond_str2 + cond_str3
        if cond_str == '000':
            continue
        
        cond_str_arr.append(cond_str)
        condm = np.matmul(currR, symmT[i])

    print(cond_str_arr)
    input()
    

    
    
def hkl_sphere_monoclinic_w_locus(radius):
    H = np.array([[0, 0, 0]], dtype=int)
    for hh in range(-radius, radius+1):
        for kk in range(0, radius+1):
            for ll in range(0, radius+1):
                if not(hh==0 and kk==0 and ll==0):
                    if math.sqrt(hh**2+kk**2+ll**2) <= radius: #sphere
                        #if not(hh < 0 and ll==0): #locus
                        H = np.append(H, np.array([[hh, kk, ll]]), axis=0)
    H = np.delete(H, 0, axis=0)
    return H
    
    
def reindex_monoclinic(H):
    #to: (-h, h), (0, k), (0, l)
    H_new = np.array([[0, 0, 0]], dtype=int)
    symm_eq = [(-1,1,-1), (1,-1,1), (-1,-1,-1)]
    for h in H:
        if h[1] < 0 or h[2] < 0:
            for eq in symm_eq:
                h_new = (h[0]*eq[0], h[1]*eq[1], h[2]*eq[2])
                if h_new[1] >= 0 and h_new[2] >= 0:
                    H_new = np.append(H_new, np.array([[h[0]*eq[0], h[1]*eq[1], h[2]*eq[2]]]), axis=0)
                    break
        else:
            if h[2] == 0 and h[0] < 0: #locus layer = hk0 and not -hk0
                H_new = np.append(H_new, np.array([[-h[0], h[1], h[2]]]), axis=0)
            else:
                H_new = np.append(H_new, np.array([[h[0], h[1], h[2]]]), axis=0)
    
    H_new = np.delete(H_new, 0, axis=0)
    return H_new

def merge_reflections(H, F):
    H_reind = reindex_monoclinic(H)
    sort_array = np.lexsort((H_reind[:, 2], H_reind[:, 1], H_reind[:, 0]))
    H_reind = H_reind[sort_array]
    F = F[sort_array]

    H_final = np.array([[0, 0, 0]])
    F_final = np.array([])
    group = [F[0]]
    H_curr = H_reind[0]

    for i in range(len(H_reind)):
        if (H_reind[i] == H_curr).all():
            group.append(F[i])
        else:
            H_final = np.append(H_final, np.array([H_curr]), axis=0)
            F_final = np.append(F_final, sum(group) / len(group))
            H_curr = H_reind[i]
            group = [F[i]]
    H_final = np.append(H_final, np.array([H_curr]), axis=0)
    F_final = np.append(F_final, sum(group) / len(group))
    H_final = np.delete(H_final, 0, axis=0)
    
    max_F_final = max(F_final)
    F_final = F_final / max_F_final
    
    return H_final, F_final
    

