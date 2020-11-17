"""main"""
# coding: utf-8
from . import create_log
from .utils import command_manager

def main():
    log_path, out_path = command_manager.controll_argument()
    create_log.execute_create_log(log_path, out_path)

if __name__ == "__main__":
    main()
