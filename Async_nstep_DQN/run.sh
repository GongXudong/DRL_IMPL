#!/bin/bash
python t_cmotp_IL.py --env_num 4 --step_len 8
python t_cmotp_IL.py --env_num 8 --step_len 4
python t_cmotp_IL.py --env_num 16 --step_len 2
python t_cmotp_IL.py --env_num 2 --step_len 16
python t_cmotp_IL.py --env_num 32 --step_len 1