# from pyneapple.models import IVIM


# def test_mono_exp_model():
#     mono_model = "( exp(-kron(b_values, abs(d1))) * (1 - (sum([]))) ) * S0"
#     mono_function = IVIM.printer(1, ["d1", "S0"])
#
#     assert mono_function == mono_model
#
#
# def test_bi_exp_model():
#     bi_model = "( exp(-kron(b_values, abs(d1))) * f1 + exp(-kron(b_values, abs(d2))) * (1 - (sum(['f1']))) ) * S0"
#     bi_function = IVIM.printer(2, ["d1", "d2", "f1", "S0"])
#
#     assert bi_function == bi_model
#
#
# def test_tri_exp_model():
#     tri_model = (
#         "( exp(-kron(b_values, abs(d1))) * f1 + exp(-kron(b_values, abs(d2))) * f2 + exp(-kron(b_values, "
#         "abs(d3))) * (1 - (sum(['f1', 'f2']))) ) * S0"
#     )
#     tri_function = IVIM.printer(3, ["d1", "d2", "d3", "f1", "f2", "S0"])
#
#     assert tri_function == tri_model
