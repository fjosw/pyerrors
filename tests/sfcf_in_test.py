import os
import sys
import inspect
import pyerrors as pe
import pyerrors.input.sfcf as sfin
import shutil
import pytest

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def build_test_environment(path, env_type, cfgs, reps):
    shutil.copytree("tests/data/sfcf_test/data_"+env_type, (path + "/data_" + env_type))
    if env_type == "o":
        for i in range(2,cfgs+1):
            shutil.copytree(path + "/data_o/test_r0/cfg1", path + "/data_o/test_r0/cfg"+str(i))
        for i in range(1,reps):
            shutil.copytree(path + "/data_o/test_r0", path + "/data_o/test_r"+str(i))
    elif env_type == "c":
        for i in range(2,cfgs+1):
            shutil.copy(path + "/data_c/data_c_r0/data_c_r0_n1", path + "/data_c/data_c_r0/data_c_r0_n"+str(i))
        for i in range(1,reps):
            os.mkdir(path + "/data_c/data_c_r"+str(i))
            for j in range(1,cfgs+1):
                shutil.copy(path + "/data_c/data_c_r0/data_c_r0_n1",path + "/data_c/data_c_r"+str(i)+"/data_c_r"+str(i)+"_n"+str(j))
    elif env_type == "a":
        for i in range(1,reps):
            for corr in ["f_1", "f_A", "F_V0"]:
                shutil.copy(path + "/data_a/data_a_r0." + corr, path + "/data_a/data_a_r" + str(i) + "." + corr)


def test_o_bb(tmp_path):
    build_test_environment(str(tmp_path), "o",5,3)
    f_1 = sfin.read_sfcf(str(tmp_path) + "/data_o", "test", "f_1",quarks="lquark lquark", wf = 0, wf2=0, version = "2.0", corr_type="bb")
    print(f_1)
    assert len(f_1) == 1
    assert list(f_1[0].shape.keys()) == ["test_|r0","test_|r1","test_|r2"]
    assert f_1[0].value == 351.1941525454502

def test_o_bi(tmp_path):
    build_test_environment(str(tmp_path), "o",5,3)
    f_A = sfin.read_sfcf(str(tmp_path) + "/data_o", "test", "f_A",quarks="lquark lquark", wf = 0, version = "2.0")
    print(f_A)
    assert len(f_A) == 3
    assert list(f_A[0].shape.keys()) == ["test_|r0","test_|r1","test_|r2"]
    assert f_A[0].value == 65.4711887279723
    assert f_A[1].value == 1.0447210336915187
    assert f_A[2].value == -41.025094911185185

def test_o_bib(tmp_path):
    build_test_environment(str(tmp_path), "o",5,3)
    f_V0 = sfin.read_sfcf(str(tmp_path) + "/data_o", "test", "F_V0",quarks="lquark lquark", wf = 0, wf2 = 0, version = "2.0", corr_type="bib")
    print(f_V0)
    assert len(f_V0) == 3
    assert list(f_V0[0].shape.keys()) == ["test_|r0","test_|r1","test_|r2"]
    assert f_V0[0] == 683.6776090085115
    assert f_V0[1] == 661.3188585582334
    assert f_V0[2] == 683.6776090081005

def test_c_bb(tmp_path):
    build_test_environment(str(tmp_path), "c",5,3)
    f_1 = sfin.read_sfcf(str(tmp_path) + "/data_c", "data_c", "f_1", quarks="lquark lquark", wf = 0, wf2=0, version = "2.0c", corr_type="bb")
    print(f_1)
    assert len(f_1) == 1
    assert list(f_1[0].shape.keys()) == ["data_c_|r0","data_c_|r1","data_c_|r2"]
    assert f_1[0].value == 351.1941525454502

def test_c_bi(tmp_path):
    build_test_environment(str(tmp_path), "c",5,3)
    f_A = sfin.read_sfcf(str(tmp_path) + "/data_c", "data_c", "f_A", quarks="lquark lquark", wf = 0, version = "2.0c")
    print(f_A)
    assert len(f_A) == 3
    assert list(f_A[0].shape.keys()) == ["data_c_|r0","data_c_|r1","data_c_|r2"]
    assert f_A[0].value == 65.4711887279723
    assert f_A[1].value == 1.0447210336915187
    assert f_A[2].value == -41.025094911185185

def test_c_bib(tmp_path):
    build_test_environment(str(tmp_path), "c",5,3)
    f_V0 = sfin.read_sfcf(str(tmp_path) + "/data_c", "data_c", "F_V0",quarks="lquark lquark", wf = 0, wf2 = 0, version = "2.0c", corr_type="bib")
    print(f_V0)
    assert len(f_V0) == 3
    assert list(f_V0[0].shape.keys()) == ["data_c_|r0","data_c_|r1","data_c_|r2"]
    assert f_V0[0] == 683.6776090085115
    assert f_V0[1] == 661.3188585582334
    assert f_V0[2] == 683.6776090081005

def test_a_bb(tmp_path):
    build_test_environment(str(tmp_path), "a",5,3)
    f_1 = sfin.read_sfcf(str(tmp_path) + "/data_a", "data_a", "f_1", quarks="lquark lquark", wf = 0, wf2=0, version = "2.0a", corr_type="bb")
    print(f_1)
    assert len(f_1) == 1
    assert list(f_1[0].shape.keys()) == ["data_a_|r0","data_a_|r1","data_a_|r2"]
    assert f_1[0].value == 351.1941525454502

def test_a_bi(tmp_path):
    build_test_environment(str(tmp_path), "a",5,3)
    f_A = sfin.read_sfcf(str(tmp_path) + "/data_a", "data_a", "f_A", quarks="lquark lquark", wf = 0, version = "2.0a")
    print(f_A)
    assert len(f_A) == 3
    assert list(f_A[0].shape.keys()) == ["data_a_|r0","data_a_|r1","data_a_|r2"]
    assert f_A[0].value == 65.4711887279723
    assert f_A[1].value == 1.0447210336915187
    assert f_A[2].value == -41.025094911185185

def test_a_bib(tmp_path):
    build_test_environment(str(tmp_path), "a",5,3)
    f_V0 = sfin.read_sfcf(str(tmp_path) + "/data_a", "data_a", "F_V0",quarks="lquark lquark", wf = 0, wf2 = 0, version = "2.0a", corr_type="bib")
    print(f_V0)
    assert len(f_V0) == 3
    assert list(f_V0[0].shape.keys()) == ["data_a_|r0","data_a_|r1","data_a_|r2"]
    assert f_V0[0] == 683.6776090085115
    assert f_V0[1] == 661.3188585582334
    assert f_V0[2] == 683.6776090081005

def test_find_corr():
    pattern = 'name      ' + "f_A" + '\nquarks    ' + "lquark lquark" + '\noffset    ' + str(0) + '\nwf        ' + str(0)
    start_read, T = sfin._find_correlator("tests/data/sfcf_test/data_c/data_c_r0/data_c_r0_n1", "2.0c", pattern, False)
    assert start_read == 21
    assert T == 3

    pattern = 'name      ' + "f_X" + '\nquarks    ' + "lquark lquark" + '\noffset    ' + str(0) + '\nwf        ' + str(0)
    with pytest.raises(ValueError):
        sfin._find_correlator("tests/data/sfcf_test/data_c/data_c_r0/data_c_r0_n1", "2.0c", pattern, False)

    pattern = 'name      ' + "f_A" + '\nquarks    ' + "lquark lquark" + '\noffset    ' + str(0) + '\nwf        ' + str(0)
    with pytest.raises(ValueError):
        sfin._find_correlator("tests/data/sfcf_test/broken_data_c/data_c_r0/data_c_r0_n1", "2.0c", pattern, False)

def test_read_compact_file(tmp_path):
    rep_path = "tests/data/sfcf_test/broken_data_c/data_c_r0/"
    config_file = "data_c_r0_n1"
    start_read = 469
    T = 3
    b2b = False
    name = "F_V0"
    im = False
    with pytest.raises(Exception):
        sfin._read_compact_file(rep_path, config_file, start_read, T, b2b, name, im)
