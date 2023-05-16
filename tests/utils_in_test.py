import pyerrors as pe


def test_sort_names():
    my_list = ['sfqcd_r1_id5', 'sfqcd_r10_id5', 'sfqcd_r7_id5', 'sfqcd_r2_id5', 'sfqcd_r2_id9', 'sfqcd_r10_id4']
    presorted_list = ['sfqcd_r1_id5', 'sfqcd_r2_id5', 'sfqcd_r2_id9', 'sfqcd_r7_id5', 'sfqcd_r10_id4', 'sfqcd_r10_id5']

    sorted_list = pe.input.utils.sort_names(my_list)
    assert (all([sorted_list[i] == presorted_list[i] for i in range(len(sorted_list))]))


def test_sort_names_only_ids():
    my_list = ['sfcf_T_id2', 'sfcf_T_id1', 'sfcf_T_id0', 'sfcf_T_id6', 'sfcf_T_id5']
    presorted_list = ['sfcf_T_id0', 'sfcf_T_id1', 'sfcf_T_id2', 'sfcf_T_id5', 'sfcf_T_id6']

    sorted_list = pe.input.utils.sort_names(my_list)
    assert (all([sorted_list[i] == presorted_list[i] for i in range(len(sorted_list))]))


def test_sort_names_only_reps():
    my_list = ['sfcf_T_r2', 'sfcf_T_r1', 'sfcf_T_r0', 'sfcf_T_r6', 'sfcf_T_r5']
    presorted_list = ['sfcf_T_r0', 'sfcf_T_r1', 'sfcf_T_r2', 'sfcf_T_r5', 'sfcf_T_r6']

    sorted_list = pe.input.utils.sort_names(my_list)
    assert (all([sorted_list[i] == presorted_list[i] for i in range(len(sorted_list))]))

def test_sort_names_fallback():
    my_list = ['xql2', 'xql1', 'xql0', 'xql6', 'xql5']
    presorted_list = ['xql0', 'xql1', 'xql2', 'xql5', 'xql6']

    sorted_list = pe.input.utils.sort_names(my_list)
    assert (all([sorted_list[i] == presorted_list[i] for i in range(len(sorted_list))]))
