""" 集計モジュールの結合テスト """
from monolish_log_viewer import create_log

def test_execute_create_log_normaldata():
    """正常系での動作確認"""
    for any_file in ["only_solver", "normal_data", "only_other", "cg_iter"]:
        # pass command line arguments
        log_path = f"./tests/test_data/{any_file}.yml"
        out_path = f"./tests/test_data/{any_file}.html"

        write_number_of_character = create_log.execute_create_log(log_path, out_path)
        print("=================")

        assert isinstance(write_number_of_character, int) is True
