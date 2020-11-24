"""viewerの生成"""
# coding: utf-8
from . import create_log, command_manager

def main():
    """ viewerの出力 """
    # コマンドライン引数取得
    log_path, out_path = command_manager.controll_argument()
    # viewerの生成
    create_log.execute_create_log(log_path, out_path)

if __name__ == "__main__":
    main()
