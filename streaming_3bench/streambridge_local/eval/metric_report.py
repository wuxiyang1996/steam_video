#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from eval.utils import *

def evaluate_ovo(path):
    ovo_bench_records = load_json(path)

    ovo_real_time_acc = {'OCR': 0, 'ACR': 0, 'ATR': 0, 'STU': 0, 'FPD': 0, 'OJR': 0}
    ovo_backward_tracing_acc = {'EPM': 0, 'ASI': 0, 'HLD': 0}
    ovo_forward_active_responding_acc = {'REC': 0, 'SSR': 0, 'CRR': 0}

    ovo_real_time_num = {'OCR': 0, 'ACR': 0, 'ATR': 0, 'STU': 0, 'FPD': 0, 'OJR': 0}
    ovo_backward_tracing_num = {'EPM': 0, 'ASI': 0, 'HLD': 0}
    ovo_forward_active_responding_num = {'REC': 0, 'SSR': 0, 'CRR': 0}

    ovo_real_time_types = ovo_real_time_acc.keys()
    ovo_backward_tracing_types = ovo_backward_tracing_acc.keys()
    ovo_forward_active_responding_types = ovo_forward_active_responding_acc.keys()

    for item in ovo_bench_records:

        if item['type'] in ovo_real_time_types or item['type'] in ovo_backward_tracing_types:
            pred = item['pred'].replace('Answer: ', '')
            pred = pred.replace('The answer is ', '')
            if pred == '':
                pred = random.choice(['A', 'B', 'C', 'D', 'E'])
            if 'A.' in pred:
                pred = 'A'
            elif 'B.' in pred:
                pred = 'B'
            elif 'C.' in pred:
                pred = 'C'
            elif 'D.' in pred:
                pred = 'D'
            elif 'E.' in pred:
                pred = 'E'

            pred = pred[0].upper()
            if pred not in ['A', 'B', 'C', 'D', 'E']:
                pred = random.choice(['A', 'B', 'C', 'D', 'E'])
            answer = item['answer'][0]
            type = item['type']
        elif item['type'] in ovo_forward_active_responding_types:
            if item['type'] in ['SSR', 'CRR']:
                if 'no' in item['pred'].lower():
                    pred = 'no'
                elif 'yes' in item['pred'].lower():
                    pred = 'yes'
                else:
                    pred = random.choice(['yes', 'no'])
                answer = item['answer'].lower()
            elif item['type'] == 'REC':
                item['pred'] = item['pred'].replace('.', '')
                if len(item['pred']) == 1:
                    pred = item['pred']
                else:
                    pred = random.choice(['0', '1', '2', '3', '4', '5', '6'])
                answer = item['answer']
            else:
                raise ValueError('Not this task')

            type = item['type']

        if type in ovo_real_time_types:
            ovo_real_time_num[type] += 1
            if pred == answer:
                ovo_real_time_acc[type] += 1
        elif type in ovo_backward_tracing_types:
            ovo_backward_tracing_num[type] += 1
            if pred == answer:
                ovo_backward_tracing_acc[type] += 1
        elif type in ovo_forward_active_responding_types:
            ovo_forward_active_responding_num[type] += 1
            if pred == answer:
                ovo_forward_active_responding_acc[type] += 1
        else:
            raise ValueError('Not this task')

    print("*********OVO-Bench-real-time***********")
    realtime_avg = 0
    realtime_count = 0
    for key in ovo_real_time_acc.keys():
        if ovo_real_time_num[key] == 0:
            print(key, ': ', 'N/A')
            continue
        key_acc = ovo_real_time_acc[key]/ovo_real_time_num[key]
        print(key, ': ', key_acc)
        realtime_avg += key_acc
        realtime_count += 1
    realtime_avg = realtime_avg/realtime_count if realtime_count else 0
    print('AVG: ', realtime_avg)
    print(ovo_real_time_num)

    print("*********OVO-Bench-backward tracing***********")
    backward_avg = 0
    backward_count = 0
    for key in ovo_backward_tracing_acc.keys():
        if ovo_backward_tracing_num[key] == 0:
            print(key, ': ', 'N/A')
            continue
        key_acc = ovo_backward_tracing_acc[key]/ovo_backward_tracing_num[key]
        print(key, ': ', key_acc)
        backward_avg += key_acc
        backward_count += 1
    backward_avg = backward_avg/backward_count if backward_count else 0
    print('AVG: ', backward_avg)
    print(ovo_backward_tracing_num)


    print("*********OVO-Bench-forward active responding***********")
    forwaed_avg = 0
    forward_count = 0
    for key in ovo_forward_active_responding_acc.keys():
        if ovo_forward_active_responding_num[key] == 0:
            print(key, ': ', 'N/A')
            continue
        key_acc = ovo_forward_active_responding_acc[key]/ovo_forward_active_responding_num[key]
        print(key, ': ', key_acc)
        forwaed_avg += key_acc
        forward_count += 1
    forwaed_avg = forwaed_avg/forward_count if forward_count else 0
    print('AVG: ', forwaed_avg)
    print(ovo_forward_active_responding_num)

    print("*********OVO-Bench-All-Average***********")
    group_avgs = []
    if realtime_count:
        group_avgs.append(realtime_avg)
    if backward_count:
        group_avgs.append(backward_avg)
    if forward_count:
        group_avgs.append(forwaed_avg)
    print(sum(group_avgs) / len(group_avgs) if group_avgs else 0)


def evaluaet_videomme(path):
    videomme_records = load_json(path)
    videomme_acc = {}
    videomme_num = {}

    for item in videomme_records:
        type = item['type']
        videomme_acc[type] = 0
        videomme_num[type] = 0

    for item in videomme_records:
        pred = item['pred'].replace('Answer: ', '')
        pred = pred.replace('The answer is ', '')
        if pred == '':
            pred = random.choice(['A', 'B', 'C', 'D', 'E'])
        if 'A.' in pred:
            pred = 'A'
        elif 'B.' in pred:
            pred = 'B'
        elif 'C.' in pred:
            pred = 'C'
        elif 'D.' in pred:
            pred = 'D'
        elif 'E.' in pred:
            pred = 'E'
        pred = pred[0].upper()
        if pred not in ['A', 'B', 'C', 'D', 'E']:
            pred = random.choice(['A', 'B', 'C', 'D', 'E'])
        answer = item['answer'][0]
        type = item['type']

        videomme_num[type] += 1
        if pred == answer:
            videomme_acc[type] += 1

    videomme_acc_all_classes = {}
    avg = 0
    for key in videomme_acc.keys():
        key_acc = videomme_acc[key]/videomme_num[key]
        videomme_acc_all_classes[key] = key_acc

    print("\n*********VideoMME***********")
    avg = 0
    for cls in videomme_acc_all_classes.keys():
        avg += videomme_acc_all_classes[cls]
    print(videomme_acc_all_classes)
    avg = avg/len(videomme_acc_all_classes.keys())
    print('AVG: ', avg)


if __name__ == "__main__":
    result_dir = os.environ.get("RESULT_DIR", "eval/results")
    evaluate_ovo(os.path.join(result_dir, 'ovo_bench_eval.json'))
    evaluaet_videomme(os.path.join(result_dir, 'videomme_eval.json'))
