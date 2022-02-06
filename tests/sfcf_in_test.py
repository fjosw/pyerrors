import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pyerrors as pe
import pyerrors.input.openQCD as qcdin
import pyerrors.input.sfcf as sfin
import shutil
import pytest

from time import sleep

n = 5
def o_test():
    for i in range(2,n+1):
        os.mkdir("data/sfcf_test/data_o/cfg"+str(i))
        shutil.copy("data/sfcf_test/data_o/cfg1/f_1","data/sfcf_test/data_o/cfg"+str(i))
        shutil.copy("data/sfcf_test/data_o/cfg1/f_A","data/sfcf_test/data_o/cfg"+str(i))
        shutil.copy("data/sfcf_test/data_o/cfg1/F_V0","data/sfcf_test/data_o/cfg"+str(i))

    o = sfin.read_sfcf("data/sfcf_test/data_o", "qcd2sf_T24L24_b3.685_k0.1394400_id0", "f_A",quarks="lquark lquark", noffset=15)#, files = ["qcd2sf_T24L24_b3.685_k0.1394400_id0_r0_n50","qcd2sf_T24L24_b3.685_k0.1394400_id0_r0_n100","qcd2sf_T24L24_b3.685_k0.1394400_id0_r0_n120","qcd2sf_T24L24_b3.685_k0.1394400_id0_r0_n140","qcd2sf_T24L24_b3.685_k0.1394400_id0_r0_n150"])


    sleep(10)
    for i in range(2,n+1):
        shutil.rmtree("data/sfcf_test/data_o/cfg"+str(i))
        
    