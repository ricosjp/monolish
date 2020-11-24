"""test_create_log
    集計モジュールの結合テスト
"""
from monolish_log_viewer import create_log

def test_execute_create_log_normaldata():
    """test_execute_create_log_normaldata
        正常データでviewerの生成の状態をみる。
        文字数を確認している
    """
    data_dict = {
        "only_solver":7013,
        "normal_data":11230,
        "only_other":1248,
        "cg_iter":123558
    }


    for file_name, word_count in data_dict.items():
        # pass command line arguments
        log_path = f"./tests/test_data/{file_name}.yml"
        out_path = f"./tests/test_data/{file_name}.html"
        write_number_of_character = create_log.execute_create_log(log_path, out_path)

        assert write_number_of_character == word_count
