action_statistic_dof = {
    "x2_normal": {
        # water flowers
        "follow_left_arm_joint_cur": {
            "min": [-3.7121],
            "delta": [7.6008],
        },
        "follow_right_arm_joint_cur": {
            "min": [-3.6176],
            "delta": [8.5015],
        },
        "follow_left_ee_cartesian_pos": {
            "min": [-0.036, -0.3241, -0.1245],
            "delta": [0.4389, 0.557, 0.479],
        },
        "follow_left_ee_rotation": {
            "min": [-1.2373, -0.1929, -1.5182],
            "delta": [2.2009, 1.5669, 2.0936],
        },
        "follow_left_gripper": {"min": [-0.1196], "delta": [4.5226]},
        "follow_right_ee_cartesian_pos": {
            "min": [-0.0326, -0.2273, -0.1377],
            "delta": [0.4574, 0.5704, 0.4743],
        },
        "follow_right_ee_rotation": {
            "min": [-1.2201, -0.2611, -0.7427],
            "delta": [2.6623, 1.6622, 2.4186],
        },
        "follow_right_gripper": {"min": [-0.1208], "delta": [4.5261]},
        "height": {"min": [-0.0001], "delta": [0.5051]},
        "head_actions": {"min": [-1.5000, -1.4167], "delta": [2.5000, 1.8879]},
        "base_velocity": {
            "min": [-0.0359, -0.084, -0.0162],
            "delta": [0.1539, 0.1848, 0.0322],
        },
    },
    "DobbE": {
        "follow_right_ee_cartesian_pos": {
            "min": [-0.6107, -0.3272, -0.4282],
            "delta": [1.2629, 1.5297, 0.8349],
        },
        "follow_right_ee_rotation": {
            "min": [-1.7378, -1.4597, -1.8712],
            "delta": [2.7031, 2.8182, 3.5921],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [0.9983]},
    },
    "RH20T": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3646, -0.2722, 0.0066],
            "delta": [0.3813, 0.5973, 0.3277],
        },
        "follow_right_ee_rotation": {
            "min": [-1.8716, -0.4398, -3.1414],
            "delta": [3.4145, 1.0225, 6.2828],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [95.0]},
    },
    "agibotworld_alpha": {
        "follow_left_ee_cartesian_pos": {
            "min": [0.4954, 0.0166, 0.1729],
            "delta": [0.3336, 0.5123, 0.9189],
        },
        "follow_left_ee_rotation": {
            "min": [-3.1064, -1.2629, -3.1238],
            "delta": [6.2127, 2.5923, 6.2496],
        },
        "follow_left_gripper": {"min": [34.6222], "delta": [86.1921]},
        "follow_right_ee_cartesian_pos": {
            "min": [0.4615, -0.5975, 0.1638],
            "delta": [0.3823, 0.5577, 0.8873],
        },
        "follow_right_ee_rotation": {
            "min": [-3.0891, -1.0739, -2.5091],
            "delta": [6.1707, 2.3074, 3.8533],
        },
        "follow_right_gripper": {"min": [34.6222], "delta": [85.7635]},
        "height": {"min": [0.0], "delta": [0.4535]},
        "head_actions": {"min": [-0.1746, 0.0523], "delta": [0.2444, 0.4713]},
    },
    "austin_buds": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3496, -0.2855, 0.0105],
            "delta": [0.3748, 0.492, 0.3116],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1405, -0.151, -0.0737],
            "delta": [6.2813, 0.3218, 0.1536],
        },
        "follow_right_gripper": {"min": [0.0076], "delta": [0.0724]},
    },
    "austin_sailor": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.387, -0.3165, 0.0244],
            "delta": [0.2999, 0.5252, 0.2308],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1402, -0.1618, -1.5918],
            "delta": [6.2804, 0.337, 2.9478],
        },
        "follow_right_gripper": {"min": [0.0005], "delta": [0.0773]},
    },
    "austin_sirius": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.0, -0.1182, 0.0],
            "delta": [0.5329, 0.3812, 0.2723],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1407, -0.1243, -1.7434],
            "delta": [6.2823, 0.1975, 1.8073],
        },
        "follow_right_gripper": {"min": [0.0334], "delta": [0.046]},
    },
    "bc_z": {
        "follow_right_ee_cartesian_pos": {
            "min": [-0.3883, -0.1116, 0.6113],
            "delta": [0.7199, 0.4288, 0.3709],
        },
        "follow_right_ee_rotation": {
            "min": [-1.056, -1.0587, -2.6295],
            "delta": [1.9142, 1.9455, 4.8064],
        },
        "follow_right_gripper": {"min": [0.2], "delta": [0.8]},
    },
    "berkeley_autolab_ur5": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3018, -0.2129, -0.1888],
            "delta": [0.3121, 0.52, 0.3107],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1396, -0.2278, 1.1413],
            "delta": [6.279, 0.454, 0.9841],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [1.0]},
    },
    "berkeley_cable_routing": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.4617, -0.28, 0.03],
            "delta": [0.1838, 0.5665, 0.1272],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1413, -0.0299, -0.7665],
            "delta": [6.2826, 0.0692, 3.322],
        },
    },
    "berkeley_fanuc_manipulation": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3718, -0.4072, 0.0184],
            "delta": [0.3483, 0.7201, 0.5229],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1399, -1.0166, -1.6988],
            "delta": [6.2802, 1.4498, 3.2074],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [1.0]},
    },
    "bridge_data_v2": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.1498, -0.2178, -0.0901],
            "delta": [0.3012, 0.469, 0.298],
        },
        "follow_right_ee_rotation": {
            "min": [-0.3279, -0.6105, -1.0578],
            "delta": [0.7378, 1.0353, 2.2552],
        },
        "follow_right_gripper": {"min": [0.0692], "delta": [0.9426]},
    },
    "dlr_edan_shared_control": {
        "follow_right_ee_cartesian_pos": {
            "min": [-0.8387, 0.1473, -0.3934],
            "delta": [0.6579, 0.6025, 1.1566],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1217, -1.5197, -2.2516],
            "delta": [6.2505, 1.5594, 4.2831],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [1.0]},
    },
    "droid": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.2667, -0.4396, -0.0472],
            "delta": [0.5159, 0.8806, 0.8331],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1374, -1.216, -2.1741],
            "delta": [6.2749, 2.1075, 4.2259],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [0.9912]},
    },
    "fmb": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3554, -0.2844, 0.0354],
            "delta": [0.336, 0.4961, 0.2943],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1404, -0.9302, -0.0599],
            "delta": [6.2807, 1.724, 1.8284],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [1.0]},
    },
    "fractal": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3242, -0.2836, 0.1405],
            "delta": [0.5518, 0.4963, 0.9328],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1308, -0.2421, -2.9685],
            "delta": [6.2609, 1.7343, 5.819],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [1.0]},
    },
    "furniture_bench": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3691, -0.181, 0.0058],
            "delta": [0.2962, 0.3582, 0.1775],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1394, -0.6121, -1.9958],
            "delta": [6.2786, 1.6114, 3.7748],
        },
        "follow_right_gripper": {"min": [0.0035], "delta": [0.0762]},
    },
    "jaco_play": {
        "follow_right_ee_cartesian_pos": {
            "min": [-0.3787, -0.6294, 0.1682],
            "delta": [0.5898, 0.3587, 0.2183],
        },
        "follow_right_ee_rotation": {
            "min": [0.9792, -0.0668, -0.0498],
            "delta": [0.0175, 0.1277, 0.0686],
        },
        "follow_right_gripper": {"min": [0.0791], "delta": [0.1033]},
    },
    "nyu_rot": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.25, -1.0, -0.2],
            "delta": [0.75, 2.0, 1.2],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1416, -3.1416, 6.2831],
            "delta": [9.4248, 4.1416, 0.0],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [1.0]},
    },
    "stanford_hydra": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.2068, -0.274, 0.1317],
            "delta": [0.4929, 0.4981, 0.4588],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1321, -0.7496, -3.0269],
            "delta": [6.2658, 1.5176, 5.8261],
        },
        "follow_right_gripper": {"min": [0.0], "delta": [0.0811]},
    },
    "stanford_kuka_multimodal": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.4781, -0.0659, 0.3424],
            "delta": [0.0868, 0.0864, 0.1863],
        },
        "follow_right_ee_rotation": {
            "min": [-3.136, -0.0521, -3.1413],
            "delta": [6.2727, 0.1109, 6.2825],
        },
        "follow_right_gripper": {"min": [-0.4713], "delta": [0.9485]},
    },
    "taco_play": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.1375, -0.4291, 0.2052],
            "delta": [0.5327, 1.0237, 0.3913],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1391, -0.6946, -1.2808],
            "delta": [6.2784, 0.8196, 3.0856],
        },
        "follow_right_gripper": {"min": [0.0001], "delta": [0.0806]},
    },
    "utaustin_mutex": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3213, -0.4734, 0.0141],
            "delta": [0.2108, 0.8471, 0.5644],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1404, -0.2202, -1.5489],
            "delta": [6.2805, 0.582, 1.9282],
        },
        "follow_right_gripper": {"min": [0.0019], "delta": [0.0738]},
    },
    "viola": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.4011, -0.2521, 0.0103],
            "delta": [0.2444, 0.4305, 0.4355],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1403, -0.2737, -1.8626],
            "delta": [6.2804, 0.4901, 2.0618],
        },
        "follow_right_gripper": {"min": [0.0002], "delta": [0.0773]},
    },
    "kuka": {
        "follow_right_ee_cartesian_pos": {
            "min": [0.3914, -0.4901, 0.0175],
            "delta": [0.3339, 0.8357, 0.9064],
        },
        "follow_right_ee_rotation": {
            "min": [-3.1416, -0.9903, -3.1416],
            "delta": [6.2832, 2.2421, 6.2832],
        },
        "follow_right_gripper": {
            "min": [0.0000],
            "delta": [1.0000],
        },
    },
    "UMI-biarm": {
        "follow_left_ee_cartesian_pos": {
            "min": [-0.2917, -0.4926, 0.0063],
            "delta": [0.9028, 0.8168, 0.3473],
        },
        "follow_left_ee_rotation": {
            "min": [-2.5309, -1.5706, -1.3309],
            "delta": [0.8758, 2.3315, 1.7076],
        },
        "follow_left_gripper": {
            "min": [0.0029],
            "delta": [0.0812],
        },
        "follow_right_ee_cartesian_pos": {
            "min": [-0.0023, -0.5191, -0.0358],
            "delta": [0.7474, 0.8668, 0.351],
        },
        "follow_right_ee_rotation": {
            "min": [-2.4945, -2.0149, -0.8088],
            "delta": [1.0941, 3.2628, 2.0018],
        },
        "follow_right_gripper": {
            "min": [0.0019],
            "delta": [0.0814],
        },
    },
    "agibotworld_beta": {
        "follow_left_ee_cartesian_pos": {
            "min": [0.4954, 0.0166, 0.1729],
            "delta": [0.3336, 0.5123, 0.9189],
        },
        "follow_left_ee_rotation": {
            "min": [-3.1064, -1.2629, -3.1238],
            "delta": [6.2127, 2.5923, 6.2496],
        },
        "follow_left_gripper": {"min": [34.6222], "delta": [86.1921]},
        "follow_right_ee_cartesian_pos": {
            "min": [0.4615, -0.5975, 0.1638],
            "delta": [0.3823, 0.5577, 0.8873],
        },
        "follow_right_ee_rotation": {
            "min": [-3.0891, -1.0739, -2.5091],
            "delta": [6.1707, 2.3074, 3.8533],
        },
        "follow_right_gripper": {"min": [34.6222], "delta": [85.7635]},
        "height": {"min": [0.0], "delta": [0.4535]},
        "head_actions": {"min": [-0.1746, 0.0523], "delta": [0.2444, 0.4713]},
    },
}
