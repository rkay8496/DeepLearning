import copy
import os
import json


def read_file():
    f = open(os.getcwd() + '/traces_evaluate_1.json', 'r')
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
            'x': False,
            'y': False,
            'hv': False,
            'vv': False,
            'a': False,
            'av': False,
            'l': False,
            'r': False,
            'act': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['x'])):
            item = {
                'No': -1,
                'x': obj['x'][i][1] * 1000,
                'y': obj['y'][i][1] * 1000,
                'hv': obj['hv'][i][1] * 1000,
                'vv': obj['vv'][i][1] * 1000,
                'a': obj['a'][i][1] * 1000,
                'av': obj['av'][i][1] * 1000,
                'l': obj['l'][i][1] * 1000,
                'r': obj['r'][i][1] * 1000,
                'act': obj['act'][i][1],
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
                    elif l['x'] == j['x'] and l['y'] == j['y'] and \
                            l['hv'] == j['hv'] and \
                            l['vv'] == j['vv'] and l['a'] == j['a'] and \
                            l['av'] == j['av'] and \
                            l['l'] == j['l'] and l['r'] == j['r'] and \
                            l['act'] == j['act']:
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
    f = open('model.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module PSD\n' \
            '\t\n' \
            '\tx : [-1500..1500];\n' \
            '\ty : [-1500..1500];\n' \
            '\thv : [-5000..5000];\n' \
            '\tvv : [-5000..5000];\n' \
            '\ta : [-3140..3140];\n' \
            '\tav : [-5000..5000];\n' \
            '\tl : [0..1000];\n' \
            '\tr : [0..1000];\n' \
            '\tact : [0..4];\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (hv = ' + str(state['hv']) + \
                ') & (vv = ' + str(state['vv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (act = ' + str(state['act']) + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(x\' = ' + str(next['x']) + ') & (y\' = ' + str(next['y']) + \
                        ') & (hv\' = ' + str(next['hv']) + \
                        ') & (vv\' = ' + str(next['vv']) + ') & (a\' = ' + str(next['a']) + \
                        ') & (av\' = ' + str(next['av']) + \
                        ') & (l\' = ' + str(next['l']) + ') & (r\' = ' + str(next['r']) + \
                        ') & (act\' = ' + str(next['act']) + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(x\' = ' + str(state['x']) + ') & (y\' = ' + str(state['y']) +  \
                    ') & (hv\' = ' + str(state['hv']) + \
                    ') & (vv\' = ' + str(state['vv']) + ') & (a\' = ' + str(state['a']) + \
                    ') & (av\' = ' + str(state['av']) + \
                    ') & (l\' = ' + str(state['l']) + ') & (r\' = ' + str(state['r']) + \
                    ') & (act\' = ' + str(state['act']) + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if len(state['next']) > 0:
            data += '((x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (hv = ' + str(state['hv']) + \
                ') & (vv = ' + str(state['vv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (act = ' + str(state['act']) + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if len(state['next']) == 1 and state['next'][0]['is_dummy']:
            data += '((x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (hv = ' + str(state['hv']) + \
                ') & (vv = ' + str(state['vv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (act = ' + str(state['act']) + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (hv = ' + str(state['hv']) + \
                ') & (vv = ' + str(state['vv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (act = ' + str(state['act']) + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(x = ' + str(state['x']) + ') & (y = ' + str(state['y']) + \
                ') & (hv = ' + str(state['hv']) + \
                ') & (vv = ' + str(state['vv']) + ') & (a = ' + str(state['a']) + \
                ') & (av = ' + str(state['av']) + \
                ') & (l = ' + str(state['l']) + ') & (r = ' + str(state['r']) + \
                ') & (act = ' + str(state['act']) + ');\n'
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
