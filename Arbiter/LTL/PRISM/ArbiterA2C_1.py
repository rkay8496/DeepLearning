import copy
import os
import json


def read_file():
    f = open(os.getcwd() + '/../ArbiterA2C_1_LTL_traces.json', 'r')
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
            'r0': False,
            'r1': False,
            'g0': False,
            'g1': False,
            'is_dummy': True
        }
        item = {}
        for i in range(len(obj['r0'])):
            item = {
                'No': -1,
                'r0': obj['r0'][i][1],
                'r1': obj['r1'][i][1],
                'g0': obj['g0'][i][1],
                'g1': obj['g1'][i][1],
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
                    if l['r0'] == j['r0'] and l['r1'] == j['r1'] and \
                            l['g0'] == j['g0'] and l['g1'] == j['g1'] and \
                            l['is_dummy'] == j['is_dummy']:
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
    f = open('./ArbiterA2C_1_LTL.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module Arbiter\n' \
            '\t\n' \
            '\tr0 : bool init false;\n' \
            '\tr1 : bool init false;\n' \
            '\tg0 : bool init false;\n' \
            '\tg1 : bool init false;\n' \
            '\tdummy : bool init false;\n' \
            '\n'
    for state in state_list:
        data += '\t[] '
        data += '(r0 = ' + str(state['r0']).lower() + ') & (r1 = ' + str(state['r1']).lower() + \
                ') & (g0 = ' + str(state['g0']).lower() + ') & (g1 = ' + str(state['g1']).lower() + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(r0\' = ' + str(next['r0']).lower() + ') & (r1\' = ' + str(next['r1']).lower() + \
                        ') & (g0\' = ' + str(next['g0']).lower() + ') & (g1\' = ' + str(next['g1']).lower() + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(r0\' = ' + str(state['r0']).lower() + ') & (r1\' = ' + str(state['r1']).lower() +  \
                    ') & (g0\' = ' + str(state['g0']).lower() + ') & (g1\' = ' + str(state['g1']).lower() + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if len(state['next']) > 0:
            data += '((r0 = ' + str(state['r0']).lower() + ') & (r1 = ' + str(state['r1']).lower() +  \
                    ') & (g0 = ' + str(state['g0']).lower() + ') & (g1 = ' + str(state['g1']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if len(state['next']) == 1 and state['next'][0]['is_dummy']:
            data += '((r0 = ' + str(state['r0']).lower() + ') & (r1 = ' + str(state['r1']).lower() +  \
                    ') & (g0 = ' + str(state['g0']).lower() + ') & (g1 = ' + str(state['g1']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(r0 = ' + str(state['r0']).lower() + ') & (r1 = ' + str(state['r1']).lower() +  \
                        ') & (g0 = ' + str(state['g0']).lower() + ') & (g1 = ' + str(state['g1']).lower() + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(r0 = ' + str(state['r0']).lower() + ') & (r1 = ' + str(state['r1']).lower() +  \
                    ') & (g0 = ' + str(state['g0']).lower() + ') & (g1 = ' + str(state['g1']).lower() + ');\n'
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
