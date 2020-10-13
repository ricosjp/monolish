#!/usr/bin/env python3
# coding: utf-8

import datetime

class Logger:
    def log_general(self, message) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] {message}")

    def log_success(self, message) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] success {message}")

    def log_error(self, message) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] error {message}")

    def dump(self, dict_list) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] dump dict")
        for one_dict in dict_list:
            print(f"{one_dict}")