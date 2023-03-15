import pyerrors as pe


def test_sort_names():
    my_list = ['sfqcd_r1_id5', 'sfqcd_r10_id5', 'sfqcd_r7_id5', 'sfqcd_r2_id5', 'sfqcd_r2_id9', 'sfqcd_r10_id4']
    presorted_list = ['sfqcd_r1_id5', 'sfqcd_r2_id5', 'sfqcd_r2_id9', 'sfqcd_r7_id5', 'sfqcd_r10_id4', 'sfqcd_r10_id5']

    sorted_list = pe.input.utils.sort_names(my_list)
    assert (all([sorted_list[i] == presorted_list[i] for i in range(len(sorted_list))]))