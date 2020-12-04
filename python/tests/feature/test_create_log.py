"""test_create_log
    集計モジュールの結合テスト
"""
from monolish_log_viewer import create_log

def test_execute_create_log_normaldata():
    """test_execute_create_log_normaldata
        正常データで表示されるhtmlが入っているかどうかみる
    """
    data_list = [
        "only_solver",
        "normal_data",
        "normal_data_order",
        # "only_other",
        "cg_iter"
        ]

    for file_name in data_list:
        # pass command line arguments
        log_path = f"./tests/test_data/{file_name}.yml"
        out_path = f"./tests/test_data/{file_name}.html"
        text_html = create_log.execute_create_log(log_path, out_path)

        assert ("<!DOCTYPE html>" in text_html) is True
        assert ('<html lang="ja">' in text_html) is True
        assert ('</html>' in text_html) is True
