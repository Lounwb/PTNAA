# -*- coding = utf-8 -*-
"""
@Time :2023/11/13 20:01
@Author: Lounwb
@File : auto.py.py
"""
import subprocess
import numpy as np

atk_list = [x for x in np.arange(15, 30, step=5)]

if __name__ == '__main__':
    # for
    # result = subprocess.run(['ls', '-l'], capture_output=True, text=True)
    #
    # print(result.stdout)
    for x in atk_list:
        # command = [
        #     'python',
        #     'NAA_bezier.py',
        #     '--r', f'{x} / 255'
        # ]
        # subprocess.run(command, check=True)
        jz = 'python verify.py --ori_path ./dataset/images/ --adv_path ./adv/NAA_bizer/ --output_file ./log2.csv'
        subprocess.run(jz, shell=True, check=True)

    print("===============Execution over============")