# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-02-24
# 参考: None
# base: None
# 功能: 从乐谱到映射的全流程检测
from ms_test import MSTester
from row_test import RowTester
from data_utils.rows_process import crop_line_detect
from data_utils.key_map import xlabel2keys


def notes_detect():
    ms_tester = MSTester()
    ms_tester.test()
    crop_line_detect(ms_tester.test_source,
                     ms_tester.gmp_logger.save_ms,
                     ms_tester.gmp_logger.save_row)

    row_tester = RowTester(ms_tester.args_creater,
                           ms_tester.gmp_logger)
    row_tester.test()
    xlabel2keys(row_tester.gmp_logger.save_row,
                row_tester.gmp_logger.save_dir,
                strain_dist={'yosabi1': 'G', 'mnzl2': 'D', 'songbie': 'E',
                             'mnzl': 'E'})


if __name__ == '__main__':
    notes_detect()
