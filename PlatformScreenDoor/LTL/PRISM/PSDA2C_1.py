import copy
import os
import json


def read_file():
    f = open(os.getcwd() + '/../WaldoA2C_1_LTL_traces.json', 'r')
    lines = f.readlines()
    f.close()
    return lines


def convert_to_traces(lines):
    converted = []
    for line in lines:
        obj = json.loads(line)
        path = []
        dummy = {
            'No': -1,
            'force': False,
            'arrived': False,
            'moving': False,
            'closed': False,
            'opened': False,
            'stuck': False,
            'obstacle': False,
            'emergency': False,
            'close': False,
            'open': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['force'])):
            item = {
                'No': -1,
                'force': obj['force'][i][1],
                'arrived': obj['arrived'][i][1],
                'moving': obj['moving'][i][1],
                'closed': obj['closed'][i][1],
                'opened': obj['opened'][i][1],
                'stuck': obj['stuck'][i][1],
                'obstacle': obj['obstacle'][i][1],
                'emergency': obj['emergency'][i][1],
                'close': obj['close'][i][1],
                'open': obj['open'][i][1],
                'is_dummy': False
            }
            path.append(item)
        if obj['aux0']:
            path.append(dummy)
            path.append(item)
        converted.append(path)
    num = 0
    find = False
    for i in converted:
        for j in i:
            if j['No'] == -1:
                j['No'] = num
                num += 1
            for k in converted:
                for l in k:
                    if l['is_dummy'] and j['is_dummy']:
                        l['No'] = j['No']
                    elif l['force'] == j['force'] and l['arrived'] == j['arrived'] and \
                            l['moving'] == j['moving'] and \
                            l['closed'] == j['closed'] and l['opened'] == j['opened'] and \
                            l['stuck'] == j['stuck'] and \
                            l['obstacle'] == j['obstacle'] and l['emergency'] == j['emergency'] and \
                            l['close'] == j['close'] and l['open'] == j['open']:
                        l['No'] = j['No']
    return converted


def calculate_probabilities(converted):
    state_list = []
    for path in converted:
        for state in path:
            state['next'] = []
            if len([s for s in state_list if s['No'] == state['No']]) == 0:
                state_list.append(state)
    for state in state_list:
        for path in converted:
            for idx, step in enumerate(path):
                if state['No'] == step['No']:
                    if idx < len(path) - 1:
                        find = False
                        for s in state['next']:
                            if s['No'] == path[idx + 1]['No']:
                                s['num_of_occurrence'] += 1
                                s['probability'] = ''
                                find = True
                                break
                            else:
                                find = False
                        if not find:
                            obj = copy.deepcopy(path[idx + 1])
                            del obj['next']
                            obj['num_of_occurrence'] = 1
                            obj['probability'] = ''
                            state['next'].append(obj)
    for state in state_list:
        total_occurrence = 0
        for s in state['next']:
            total_occurrence += s['num_of_occurrence']
        for s in state['next']:
            s['probability'] = str(s['num_of_occurrence']) + '/' + str(total_occurrence)
    for state in state_list:
        print(state)
    return state_list


def generate_prism_model(state_list):
    f = open('./PSDA2C_1_LTL.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module PSD\n' \
            '\t\n' \
            '\tforce : bool init false;\n' \
            '\tarrived : bool init true;\n' \
            '\tmoving : bool init false;\n' \
            '\tclosed : bool init true;\n' \
            '\topened : bool init false;\n' \
            '\tstuck : bool init false;\n' \
            '\tobstacle : bool init false;\n' \
            '\temergency : bool init false;\n' \
            '\tclose : bool init false;\n' \
            '\topen : bool init true;\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(force = ' + str(state['force']).lower() + ') & (arrived = ' + str(state['arrived']).lower() + \
                ') & (moving = ' + str(state['moving']).lower() + \
                ') & (closed = ' + str(state['closed']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                ') & (stuck = ' + str(state['stuck']).lower() + \
                ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (emergency = ' + str(state['emergency']).lower() + \
                ') & (close = ' + str(state['close']).lower() + ') & (open = ' + str(state['open']).lower() + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(force\' = ' + str(next['force']).lower() + ') & (arrived\' = ' + str(next['arrived']).lower() + \
                        ') & (moving\' = ' + str(next['moving']).lower() + \
                        ') & (closed\' = ' + str(next['closed']).lower() + ') & (opened\' = ' + str(next['opened']).lower() + \
                        ') & (stuck\' = ' + str(next['stuck']).lower() + \
                        ') & (obstacle\' = ' + str(next['obstacle']).lower() + ') & (emergency\' = ' + str(next['emergency']).lower() + \
                        ') & (close\' = ' + str(next['close']).lower() + ') & (open\' = ' + str(next['open']).lower() + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(force\' = ' + str(state['force']).lower() + ') & (arrived\' = ' + str(state['arrived']).lower() +  \
                    ') & (moving\' = ' + str(state['moving']).lower() + \
                    ') & (closed\' = ' + str(state['closed']).lower() + ') & (opened\' = ' + str(state['opened']).lower() + \
                    ') & (stuck\' = ' + str(state['stuck']).lower() + \
                    ') & (obstacle\' = ' + str(state['obstacle']).lower() + ') & (emergency\' = ' + str(state['emergency']).lower() + \
                    ') & (close\' = ' + str(state['close']).lower() + ') & (open\' = ' + str(state['open']).lower() + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if len(state['next']) > 0:
            data += '((force = ' + str(state['force']).lower() + ') & (arrived = ' + str(state['arrived']).lower() + \
                ') & (moving = ' + str(state['moving']).lower() + \
                ') & (closed = ' + str(state['closed']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                ') & (stuck = ' + str(state['stuck']).lower() + \
                ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (emergency = ' + str(state['emergency']).lower() + \
                ') & (close = ' + str(state['close']).lower() + ') & (open = ' + str(state['open']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if len(state['next']) == 1 and state['next'][0]['is_dummy']:
            data += '((force = ' + str(state['force']).lower() + ') & (arrived = ' + str(state['arrived']).lower() + \
                ') & (moving = ' + str(state['moving']).lower() + \
                ') & (closed = ' + str(state['closed']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                ') & (stuck = ' + str(state['stuck']).lower() + \
                ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (emergency = ' + str(state['emergency']).lower() + \
                ') & (close = ' + str(state['close']).lower() + ') & (open = ' + str(state['open']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(force = ' + str(state['force']).lower() + ') & (arrived = ' + str(state['arrived']).lower() + \
                ') & (moving = ' + str(state['moving']).lower() + \
                ') & (closed = ' + str(state['closed']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                ') & (stuck = ' + str(state['stuck']).lower() + \
                ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (emergency = ' + str(state['emergency']).lower() + \
                ') & (close = ' + str(state['close']).lower() + ') & (open = ' + str(state['open']).lower() + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(force = ' + str(state['force']).lower() + ') & (arrived = ' + str(state['arrived']).lower() + \
                ') & (moving = ' + str(state['moving']).lower() + \
                ') & (closed = ' + str(state['closed']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                ') & (stuck = ' + str(state['stuck']).lower() + \
                ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (emergency = ' + str(state['emergency']).lower() + \
                ') & (close = ' + str(state['close']).lower() + ') & (open = ' + str(state['open']).lower() + ');\n'
    data += '\n'
    data += 'rewards "step"\n'
    data += '\t[] true : 1;\n'
    data += 'endrewards'
    print(data)
    f.write(data)
    f.close


lines = read_file()
converted = convert_to_traces(lines)
state_list = calculate_probabilities(converted)
generate_prism_model(state_list)
