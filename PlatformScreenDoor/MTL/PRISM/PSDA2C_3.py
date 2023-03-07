import copy
import os
import json


def read_file():
    f = open(os.getcwd() + '/../PSDA2C_3_MTL_traces.json', 'r')
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
            'standstill': False,
            'getonandoff': False,
            'pickup': False,
            'closed': False,
            'partially': False,
            'opened': False,
            'obstacle': False,
            'idle': False,
            'open': False,
            'close': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['standstill'])):
            item = {
                'No': -1,
                'standstill': obj['standstill'][i][1],
                'getonandoff': obj['getonandoff'][i][1],
                'pickup': obj['pickup'][i][1],
                'closed': obj['closed'][i][1],
                'partially': obj['partially'][i][1],
                'opened': obj['opened'][i][1],
                'obstacle': obj['obstacle'][i][1],
                'idle': obj['idle'][i][1],
                'open': obj['open'][i][1],
                'close': obj['close'][i][1],
                'is_dummy': False
            }
            path.append(item)
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
                    elif l['standstill'] == j['standstill'] and l['getonandoff'] == j['getonandoff'] and \
                            l['pickup'] == j['pickup'] and l['closed'] == j['closed'] and \
                            l['partially'] == j['partially'] and l['opened'] == j['opened'] and \
                            l['obstacle'] == j['obstacle'] and l['idle'] == j['idle'] and \
                            l['open'] == j['open'] and l['close'] == j['close']:
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
    f = open('./PSDA2C_3_MTL.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module PSD\n' \
            '\t\n' \
            '\tstandstill : bool init true;\n' \
            '\tgetonandoff : bool init false;\n' \
            '\tpickup : bool init false;\n' \
            '\tclosed : bool init true;\n' \
            '\tpartially : bool init false;\n' \
            '\topened : bool init false;\n' \
            '\tobstacle : bool init false;\n' \
            '\tidle : bool init true;\n' \
            '\topen : bool init false;\n' \
            '\tclose : bool init false;\n' \
            '\n'
    for state in state_list:
        data += '\t[] '
        data += '(standstill = ' + str(state['standstill']).lower() + ') & (getonandoff = ' + str(state['getonandoff']).lower() + \
                ') & (pickup = ' + str(state['pickup']).lower() + ') & (closed = ' + str(state['closed']).lower() + \
                ') & (partially = ' + str(state['partially']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (idle = ' + str(state['idle']).lower() + \
                ') & (open = ' + str(state['open']).lower() + ') & (close = ' + str(state['close']).lower() + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(standstill\' = ' + str(next['standstill']).lower() + ') & (getonandoff\' = ' + str(next['getonandoff']).lower() + \
                        ') & (pickup\' = ' + str(next['pickup']).lower() + ') & (closed\' = ' + str(next['closed']).lower() + \
                        ') & (partially\' = ' + str(next['partially']).lower() + ') & (opened\' = ' + str(next['opened']).lower() + \
                        ') & (obstacle\' = ' + str(next['obstacle']).lower() + ') & (idle\' = ' + str(next['idle']).lower() + \
                        ') & (open\' = ' + str(next['open']).lower() + ') & (close\' = ' + str(next['close']).lower() + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(standstill\' = ' + str(state['standstill']).lower() + ') & (getonandoff\' = ' + str(state['getonandoff']).lower() +  \
                    ') & (pickup\' = ' + str(state['pickup']).lower() + ') & (closed\' = ' + str(state['closed']).lower() + \
                    ') & (partially\' = ' + str(state['partially']).lower() + ') & (opened\' = ' + str(state['opened']).lower() + \
                    ') & (obstacle\' = ' + str(state['obstacle']).lower() + ') & (idle\' = ' + str(state['idle']).lower() + \
                    ') & (open\' = ' + str(state['open']).lower() + ') & (close\' = ' + str(state['close']).lower() + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if len(state['next']) > 0:
            data += '((standstill = ' + str(state['standstill']).lower() + ') & (getonandoff = ' + str(state['getonandoff']).lower() +  \
                    ') & (pickup = ' + str(state['pickup']).lower() + ') & (closed = ' + str(state['closed']).lower() + \
                    ') & (partially = ' + str(state['partially']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                    ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (idle = ' + str(state['idle']).lower() + \
                    ') & (open = ' + str(state['open']).lower() + ') & (close = ' + str(state['close']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if len(state['next']) == 1 and state['next'][0]['is_dummy']:
            data += '((standstill = ' + str(state['standstill']).lower() + ') & (getonandoff = ' + str(state['getonandoff']).lower() +  \
                    ') & (pickup = ' + str(state['pickup']).lower() + ') & (closed = ' + str(state['closed']).lower() + \
                    ') & (partially = ' + str(state['partially']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                    ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (idle = ' + str(state['idle']).lower() + \
                    ') & (open = ' + str(state['open']).lower() + ') & (close = ' + str(state['close']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(standstill = ' + str(state['standstill']).lower() + ') & (getonandoff = ' + str(state['getonandoff']).lower() +  \
                        ') & (pickup = ' + str(state['pickup']).lower() + ') & (closed = ' + str(state['closed']).lower() + \
                        ') & (partially = ' + str(state['partially']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                        ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (idle = ' + str(state['idle']).lower() + \
                        ') & (open = ' + str(state['open']).lower() + ') & (close = ' + str(state['close']).lower() + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(standstill = ' + str(state['standstill']).lower() + ') & (getonandoff = ' + str(state['getonandoff']).lower() +  \
                    ') & (pickup = ' + str(state['pickup']).lower() + ') & (closed = ' + str(state['closed']).lower() + \
                    ') & (partially = ' + str(state['partially']).lower() + ') & (opened = ' + str(state['opened']).lower() + \
                    ') & (obstacle = ' + str(state['obstacle']).lower() + ') & (idle = ' + str(state['idle']).lower() + \
                    ') & (open = ' + str(state['open']).lower() + ') & (close = ' + str(state['close']).lower() + ');\n'
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
