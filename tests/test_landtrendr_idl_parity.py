"""Parity of zeit's LandTrendr against the original LandTrendr-2012 IDL code.

Every expected output below was produced by the original IDL source run under
GDL on the same inputs, and covers each branch of tbcd_v2.pro's control flow:
the primary (find_best_trace) ladder, the joint-fit fallback ladder, the
non-significant flat line, both modifier orientations, missing years in the
interior and at either edge (including the edge-padding vertex removal), and
best_model_proportion above 1.
"""
import numpy as np
import pytest
from zeit._landtrendr import run_landtrendr

# Reference outputs of the ORIGINAL LandTrendr-2012 IDL code (fit_trajectory_v2.pro /
# tbcd_v2.pro, KennedyResearch/LandTrendr-2012), produced by running it under GDL
# 1.1.2 on exactly these inputs. idl_values are what the original returns:
# modifier-space vertex values truncated to integers (its vertvals is an intarr).
# None marks a missing year (not in the original's `goods`).
IDL_CASES = [
    dict(
        name='flat_line',
        years=(1985, 2020),
        values=[726.2956, 732.5028, 712.9148, 745.7694, 737.0514, 734.0427, 730.5527, 739.1198, 703.7411, 733.8347, 733.6622, 697.0024, 729.7071, 720.4675, 736.4266, 717.5502, 724.0582, 729.6425, 710.7888, 733.5976, 722.717, 731.9565, 716.27, 741.1384, 733.6998, 721.5871, 734.1135, 743.8225, 752.0409, 700.0048, 737.998, 731.5326, 722.8887, 708.7721, 743.7828, 704.8274],
        params={'max_segments': 4, 'modifier': 1.0, 'best_model_proportion': 0.75, 'spike_threshold': 1.0, 'recovery_threshold': 0.25, 'pval_threshold': 0.05},
        idl_years=[1985, 2020],
        idl_values=[728.0, 728.0],
    ),
    dict(
        name='multi_vertex_gain_orientation',
        years=(1985, 2020),
        values=[609.5175, 592.3151, 618.13, 600.5902, 632.6796, 638.6053, 623.5253, 623.7462, 617.1508, 601.1737, 609.5441, 599.76, 625.8682, 614.8215, 631.5796, 575.1478, 640.6352, 591.8696, 592.7091, 608.5708, 597.7586, 623.0062, 597.44, 582.8053, 589.4141, 653.1689, 615.7122, 579.9082, 614.6521, 568.4319, 667.2798, 569.3617, 628.55, 630.4667, 571.562, 623.2766],
        params={'max_segments': 5, 'modifier': 1.0, 'best_model_proportion': 0.75, 'spike_threshold': 0.75, 'recovery_threshold': 0.5, 'pval_threshold': 0.1},
        idl_years=[1985, 2009, 2010, 2016, 2017, 2020],
        idl_values=[621.0, 598.0, 653.0, 554.0, 628.0, 621.0],
    ),
    dict(
        name='multi_vertex_loss_orientation',
        years=(1985, 2020),
        values=[555.9175, 605.7424, 619.3882, 577.7847, 609.309, 559.8927, 596.2948, 562.3247, 600.382, 560.0777, 545.0785, 598.85, 632.434, 581.2438, 606.5558, 593.7838, 586.0754, 602.743, 620.9751, 569.8919, 601.8289, 597.9454, 566.5328, 620.4834, 626.4242, 574.9044, 622.6703, 592.8151, 588.9203, 618.5083, 595.8559, 597.5555, 618.7608, 614.6874, 607.674, 566.9552],
        params={'max_segments': 6, 'modifier': -1.0, 'best_model_proportion': 1.0, 'spike_threshold': 0.9, 'recovery_threshold': 1.0, 'pval_threshold': 0.05},
        idl_years=[1985, 1987, 1995, 1997, 1998, 2020],
        idl_values=[-565.0, -605.0, -544.0, -636.0, -596.0, -603.0],
    ),
    dict(
        name='first_year_missing_full_model',
        years=(1985, 2020),
        values=[None, 715.3176, 688.3269, 745.0381, 763.6005, 726.585, 767.1634, 744.1335, 776.8836, 791.4462, 799.7343, 803.9011, 782.6424, 824.9707, None, 823.2768, 836.8975, None, 801.1434, 850.7959, 815.7328, 826.3541, 826.7576, 877.5131, 859.373, 816.0284, 891.0249, 927.5578, 926.5773, 876.6352, 897.0409, 912.4551, 926.7753, 912.0706, 891.0463, 939.5592],
        params={'max_segments': 6, 'modifier': -1.0, 'best_model_proportion': 1.0, 'spike_threshold': 0.9, 'recovery_threshold': 1.0, 'pval_threshold': 0.05},
        idl_years=[1985, 1998, 2007, 2008, 2010, 2012, 2020],
        idl_values=[-715.0, -817.0, -833.0, -877.0, -821.0, -934.0, -901.0],
    ),
    dict(
        name='last_year_missing_full_model',
        years=(1985, 2020),
        values=[582.6163, 580.4656, 558.1306, 564.3182, 596.698, 583.8596, 560.732, 575.89, 578.7228, 574.8601, 564.8189, 599.9368, 591.1008, 580.4076, 560.6686, 577.2352, 587.5044, 602.2747, None, 608.7294, None, 583.0547, 552.9985, 579.2167, 575.7435, None, 606.489, 624.1021, None, 587.7492, 643.0131, 613.0661, 605.6198, 607.0152, 591.6681, None],
        params={'max_segments': 6, 'modifier': 1.0, 'best_model_proportion': 1.25, 'spike_threshold': 0.9, 'recovery_threshold': 0.25, 'pval_threshold': 0.05},
        idl_years=[1985, 2009, 2012, 2014, 2015, 2019, 2020],
        idl_values=[575.0, 585.0, 621.0, 587.0, 643.0, 587.0, 587.0],
    ),
    dict(
        name='both_ends_missing',
        years=(1985, 2020),
        values=[None, 634.4953, 596.2172, 677.5983, 617.688, 653.5908, 657.8688, None, 628.5879, 710.6833, 698.8905, 705.6372, 723.0516, 669.1026, 607.5962, 707.7345, 738.5861, 737.9804, 694.274, 710.2066, 724.2812, 669.558, 704.8773, 709.2947, 774.6711, 716.0038, 781.4495, 741.6524, 698.1558, 810.1588, 724.9138, 772.7673, 806.6347, 747.7906, 731.9752, None],
        params={'max_segments': 6, 'modifier': 1.0, 'best_model_proportion': 1.0, 'spike_threshold': 0.9, 'recovery_threshold': 1.0, 'pval_threshold': 0.05},
        idl_years=[1985, 1986, 2019, 2020],
        idl_values=[636.0, 636.0, 768.0, 768.0],
    ),
    dict(
        name='interior_gaps',
        years=(1985, 2020),
        values=[586.1604, 602.3309, 609.4211, 592.859, 587.5307, 596.4113, 567.8114, 589.5542, 577.4301, 617.5597, None, 591.4109, 587.5057, 560.5131, 598.0577, 608.7189, 581.289, 597.7157, 598.1709, 571.759, 600.6508, 605.525, 592.4241, 589.0316, 598.9319, 571.8448, 590.8515, 602.0303, 575.3148, None, 584.9017, 565.5628, 588.5365, 578.4616, 593.48, 576.1324],
        params={'max_segments': 5, 'modifier': -1.0, 'best_model_proportion': 0.75, 'spike_threshold': 0.75, 'recovery_threshold': 0.5, 'pval_threshold': 0.1},
        idl_years=[1985, 1993, 2006, 2020],
        idl_values=[-599.0, -582.0, -602.0, -574.0],
    ),
    dict(
        name='bmp_above_one',
        years=(1985, 2020),
        values=[622.1832, 577.0557, 621.9715, 621.3505, 627.0254, 628.0076, 627.3669, 651.2563, 643.3532, 649.6872, 630.4505, 661.6815, 652.5055, 606.1072, 675.4465, 660.4498, 694.3029, 670.7263, 676.249, 687.8469, 686.064, 678.9508, 648.8754, 744.35, 661.9797, 697.0092, 725.9251, 722.893, 731.4622, 740.4012, 756.7051, 736.2102, 805.8164, 710.8819, 746.0003, 735.0274],
        params={'max_segments': 6, 'modifier': -1.0, 'best_model_proportion': 1.25, 'spike_threshold': 0.9, 'recovery_threshold': 0.25, 'pval_threshold': 0.05},
        idl_years=[1985, 1999, 2016, 2018, 2020],
        idl_values=[-619.0, -652.0, -736.0, -710.0, -744.0],
    ),
    dict(
        name='no_desawtooth',
        years=(1985, 2020),
        values=[596.6114, 606.787, 601.1586, 581.4112, 598.698, 589.5918, 611.1607, 584.1471, 610.608, 583.3905, 608.6965, 590.792, 587.5489, 589.1893, 587.9796, 591.2848, 592.6744, 567.1706, 554.7963, 536.6365, 559.8538, 570.377, 563.8253, 558.599, 551.1152, 545.5917, 547.2817, 552.5034, 526.1717, 551.7226, 525.0135, 537.2416, 529.6362, 530.3066, 520.4464, 516.953],
        params={'max_segments': 4, 'modifier': -1.0, 'best_model_proportion': 0.75, 'spike_threshold': 1.0, 'recovery_threshold': 0.25, 'pval_threshold': 0.05},
        idl_years=[1985, 1993, 1995, 2004, 2020],
        idl_values=[-596.0, -598.0, -600.0, -556.0, -525.0],
    ),
    dict(
        name='loose_recovery_threshold',
        years=(1985, 2020),
        values=[718.8755, 738.5127, 718.0147, 646.8215, 714.7553, 730.2278, 753.6738, 742.1284, 784.8096, 762.1664, 762.2379, 754.773, 703.2111, 756.2393, 752.3867, 728.374, 755.7566, 753.7178, 735.2718, 744.949, 743.5687, 756.7396, 743.5299, 767.6246, 712.9104, 752.3025, 721.9231, 751.3522, 776.1438, 766.602, 765.3975, 771.3501, 807.6433, 748.0321, 795.6525, 742.3659],
        params={'max_segments': 6, 'modifier': 1.0, 'best_model_proportion': 1.0, 'spike_threshold': 0.9, 'recovery_threshold': 1.0, 'pval_threshold': 0.05},
        idl_years=[1985, 2000, 2009, 2017, 2020],
        idl_values=[720.0, 762.0, 733.0, 792.0, 732.0],
    ),
    dict(
        name='f7_fallback_path',
        years=(1990, 2019),
        values=[717.4605, 701.5778, 703.0987, 718.7766, 695.6588, 688.7065, 694.6172, 705.428, 681.2109, 707.6497, 693.5347, 713.7738, 671.2689, 412.0467, 456.9507, 522.9765, 591.5662, 642.1464, 696.9595, 698.0898, 702.4407, 687.8976, 708.4822, 707.952, 704.6205, 706.6784, 703.557, 724.4209, 698.9549, 696.315],
        params={'max_segments': 6, 'modifier': 1.0, 'best_model_proportion': 0.75, 'spike_threshold': 0.9, 'recovery_threshold': 0.25, 'pval_threshold': 0.05},
        idl_years=[1990, 2003, 2019],
        idl_values=[744.0, 613.0, 732.0],
    ),
    dict(
        name='f6_disturbance_recovery',
        years=(1990, 2019),
        values=[687.5135, 705.6964, 698.427, 678.0291, 711.1396, 692.74, 693.5932, 687.163, 692.1486, 705.1347, 697.7291, 703.9439, 704.3431, 715.8479, 695.8866, 682.2777, 501.9066, 520.2721, 572.7751, 599.1505, 628.1263, 669.0353, 723.4122, 724.9238, 700.8326, 701.9223, 712.9149, 689.8521, 703.9968, 699.6896],
        params={'max_segments': 6, 'modifier': 1.0, 'best_model_proportion': 0.75, 'spike_threshold': 0.9, 'recovery_threshold': 0.25, 'pval_threshold': 0.05},
        idl_years=[1990, 2010, 2012, 2019],
        idl_values=[725.0, 615.0, 723.0, 693.0],
    ),
    dict(
        name='loss_with_recovery',
        years=(1985, 2020),
        values=[708.3524, 722.118, 727.0542, 665.8677, 783.3147, 721.7052, 722.4825, 668.6337, 735.2206, 729.6909, 708.6687, 725.6597, 754.4254, 710.0925, 694.6232, 756.3938, 727.1824, 715.1042, 783.4792, 724.5404, 715.4029, 727.5377, 709.1424, 726.095, 724.2859, 702.0642, 712.9505, 728.5972, 730.5525, 481.0243, 529.0595, 556.7495, 606.5882, 624.315, 670.173, 748.3006],
        params={'max_segments': 5, 'modifier': -1.0, 'best_model_proportion': 0.75, 'spike_threshold': 0.75, 'recovery_threshold': 0.5, 'pval_threshold': 0.1},
        idl_years=[1985, 1987, 1989, 2013, 2014, 2020],
        idl_values=[-709.0, -728.0, -747.0, -712.0, -481.0, -724.0],
    ),
    dict(
        name='first_segment_regression_choice',
        years=(2000, 2013),
        values=[500.0, 540.0, 460.0, 530.0, 470.0, 500.0, 300.0, 100.0, 110.0, 90.0, 100.0, 110.0, 90.0, 100.0],
        params={'max_segments': 3, 'modifier': -1.0, 'best_model_proportion': 0.75, 'spike_threshold': 0.9, 'recovery_threshold': 0.25, 'pval_threshold': 0.05},
        idl_years=[2000, 2005, 2007, 2013],
        idl_values=[-525.0, -496.0, -100.0, -98.0],
    ),
]


@pytest.mark.parametrize("case", IDL_CASES, ids=[c["name"] for c in IDL_CASES])
def test_matches_original_idl(case):
    years = np.arange(case["years"][0], case["years"][1] + 1)
    values = np.array([np.nan if v is None else v for v in case["values"]])
    params = case["params"]

    vertices = run_landtrendr(years, values, vertex_count_overshoot=3, min_observations_needed=6, **params)

    assert [v["year"] for v in vertices] == case["idl_years"]
    # The original truncates modifier-space values to integers and computes in
    # float32; zeit keeps full double precision in the original scale.
    got = np.trunc(np.array([v["value"] for v in vertices]) * params["modifier"])
    np.testing.assert_allclose(got, case["idl_values"], atol=1.0)
