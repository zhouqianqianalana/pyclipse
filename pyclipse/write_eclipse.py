import numpy as np
from pathlib import Path
import shutil

class Writer:

    def __init__(self, datafile_path, output_dirpath=None, remove_after=False):
        self.datafile_path = Path(datafile_path)
        self.templates_dirpath = self.datafile_path.parent

        if output_dirpath is None:
            self.output_dirpath = self.templates_dirpath
        else:
            self.output_dirpath = Path(output_dirpath)
            if self.output_dirpath != self.templates_dirpath:
                if not self.output_dirpath.exists():
                    self.output_dirpath.mkdir(parents=True)
                shutil.copytree(self.templates_dirpath, self.output_dirpath)

        self.remove_after = remove_after


    def write_coord(self, res):
        with open(self.output_dirpath/'COORD', 'w') as f:
            f.write('COORD                                   -- Generated : Petrel\n ')
            val_in_line = 0
            for j in range(res.X.shape[1]-1,-1,-1):  
                for i in range(res.X.shape[0]):
                    if val_in_line == 2:
                        f.write("\n ")
                        val_in_line = 0
                    val_in_line += 1         
                    f.write(" {:.7g} {:.7g} {:.7g} {:.7g} {:.7g} {:.7g}".format(res.X[i,j], -res.Y[i,j], res.zz[0][i,j],
                                                                                res.X[i,j], -res.Y[i,j], res.zz[-1][i,j]))
            f.write(" /\n")


    def write_zcorn(self, res):
        with open(self.output_dirpath/'ZCORN', 'w') as f:
            f.write('ZCORN                                   -- Generated : Petrel\n ')
            val_in_line = 0
            k_array = np.arange(res.nz+1)
            k_array = np.concatenate(([k_array[0]], np.repeat(k_array[1:-1], 2), [k_array[-1]]))
            for k in k_array:
                for j in range(res.X.shape[1]-1,-1,-1):
                    if val_in_line == 10:
                        f.write("\n ")
                        val_in_line = 0
                    val_in_line += 1
                    if j==0 or j==res.X.shape[1]-1:
                        if k==0:
                            f.write(" {}*{:.7g}".format(2*res.nx,res.zz[0][0,j]))
                        elif k==res.nz:
                            f.write(" {}*{:.7g}".format(2*res.nx,res.zz[1][0,j]))
                        else:
                            f.write(" {}*{:.7g}".format(2*res.nx,res.zz[0][0,j] + k*(res.zz[1][0,j]-res.zz[0][0,j])/res.nz))
                    else:
                        if k==0:
                            f.write(" {}*{:.7g}".format(4*res.nx,res.zz[0][0,j]))
                        elif k==res.nz:
                            f.write(" {}*{:.7g}".format(4*res.nx,res.zz[1][0,j]))
                        else:
                            f.write(" {}*{:.7g}".format(4*res.nx,res.zz[0][0,j]+k*(res.zz[1][0,j]-res.zz[0][0,j])/res.nz))
            f.write("/\n")


    def write_poro(self, res):
        with open(self.output_dirpath/'PORO', 'w') as f:
            f.write('PORO                                   -- Generated : Petrel\n-- Property name in Petrel : Porosity\n ')
            val_in_line = 0
            for k in np.arange(res.nz):
                for j in np.arange(res.ny):
                    for i in np.arange(res.nx):
                        if val_in_line == 14:
                            f.write("\n ")
                            val_in_line = 0
                        val_in_line += 1
                        f.write(' %.6f' % abs(res.poro_mat[i, res.ny-j-1, res.nz-k-1]))
            f.write(" /\n")
            

    def write_permx(self, res):
        with open(self.output_dirpath/'PERMX', 'w') as f:
            f.write('PERMX                                   -- Generated : Petrel\n-- Property name in Petrel : PERMX\n ')
            val_in_line = 0
            for k in np.arange(res.nz):
                for j in np.arange(res.ny):
                    for i in np.arange(res.nx):
                        if val_in_line == 14:
                            f.write("\n ")
                            val_in_line = 0
                        val_in_line += 1
                        if abs(res.perm_mat[i,res.ny-j-1,res.nz-k-1]) == 0.0:
                            f.write(" 0.00000")
                        else:
                            f.write(" {:.6g}".format(abs(res.perm_mat[i, res.ny-j-1, res.nz-k-1])))
            f.write(" /\n")
            

    def write_permy(self, res):
        with open(self.output_dirpath/'PERMY', 'w') as f:
            f.write('PERMY                                   -- Generated : Petrel\n-- Property name in Petrel : PERMY\n ')
            val_in_line = 0
            for k in np.arange(res.nz):
                for j in np.arange(res.ny):
                    for i in np.arange(res.nx):
                        if val_in_line == 14:
                            f.write("\n ")
                            val_in_line = 0
                        val_in_line += 1
                        if abs(res.perm_mat[i,res.ny-j-1,res.nz-k-1]) == 0.0:
                            f.write(" 0.00000")
                        else:
                            f.write(" {:.6g}".format(abs(res.perm_mat[i, res.ny-j-1, res.nz-k-1])))
            f.write(" /\n")
            

    def write_permz(self, res):
        with open(self.output_dirpath/'PERMZ', 'w') as f:
            f.write('PERMZ                                   -- Generated : Petrel\n-- Property name in Petrel : PERMZ\n ')
            val_in_line = 0
            for k in np.arange(res.nz):
                for j in np.arange(res.ny):
                    for i in np.arange(res.nx):
                        if val_in_line == 14:
                            f.write("\n ")
                            val_in_line = 0
                        val_in_line += 1
                        if abs(res.perm_mat[i,res.ny-j-1,res.nz-k-1]) == 0.0:
                            f.write(" 0.00000")
                        else:
                            f.write(" {:.6g}".format(abs(res.kzkx*res.perm_mat[i, res.ny-j-1, res.nz-k-1])))
            f.write(" /\n")
                       

    def write_actnum(self, res): 
        with open(self.output_dirpath/'ACTNUM', 'w') as f:
            f.write('ACTNUM                                   -- Generated : Petrel\n-- Property name in Petrel : ACTNUM\n ')
            val_in_line = 0
            for k in np.arange(res.nz):
                for j in np.arange(res.ny):
                    for i in np.arange(res.nx):
                        if val_in_line == 64:
                            f.write("\n ")
                            val_in_line = 0
                        val_in_line += 1
                        f.write(' %i' % (int(res.active[i, res.ny-j-1, res.nz-k-1])))
            f.write(" /\n")

