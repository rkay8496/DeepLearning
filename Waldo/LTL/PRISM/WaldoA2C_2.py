import copy
import os
import json


def read_file():
    f = open(os.getcwd() + '/../WaldoA2C_2_LTL_traces.json', 'r')
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
            'ckbby': False,
            'r1': False,
            'r2': False,
            'r3': False,
            'r4': False,
            'r5': False,
            'r6': False,
            'r7': False,
            'r8': False,
            'r9': False,
            'r10': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['ckbby'])):
            item = {
                'No': -1,
                'ckbby': obj['ckbby'][i][1],
                'r1': obj['r1'][i][1],
                'r2': obj['r2'][i][1],
                'r3': obj['r3'][i][1],
                'r4': obj['r4'][i][1],
                'r5': obj['r5'][i][1],
                'r6': obj['r6'][i][1],
                'r7': obj['r7'][i][1],
                'r8': obj['r8'][i][1],
                'r9': obj['r9'][i][1],
                'r10': obj['r10'][i][1],
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
                    elif l['ckbby'] == j['ckbby'] and l['r1'] == j['r1'] and \
                            l['r2'] == j['r2'] and \
                            l['r3'] == j['r3'] and l['r4'] == j['r4'] and \
                            l['r5'] == j['r5'] and l['r6'] == j['r6'] and \
                            l['r7'] == j['r7'] and l['r8'] == j['r8'] and \
                            l['r9'] == j['r9'] and l['r10'] == j['r10']:
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
    f = open('./WaldoA2C_2_LTL.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module Waldo\n' \
            '\t\n' \
            '\tckbby : bool init true;\n' \
            '\tr1 : bool init true;\n' \
            '\tr2 : bool init false;\n' \
            '\tr3 : bool init false;\n' \
            '\tr4 : bool init false;\n' \
            '\tr5 : bool init false;\n' \
            '\tr6 : bool init false;\n' \
            '\tr7 : bool init false;\n' \
            '\tr8 : bool init false;\n' \
            '\tr9 : bool init false;\n' \
            '\tr10 : bool init false;\n' \
           '\n'
    for state in state_list:
        data += '\t[] '
        data += '(ckbby = ' + str(state['ckbby']).lower() + ') & (r1 = ' + str(state['r1']).lower() + \
                ') & (r2 = ' + str(state['r2']).lower() + \
                ') & (r3 = ' + str(state['r3']).lower() + ') & (r4 = ' + str(state['r4']).lower() + \
                ') & (r5 = ' + str(state['r5']).lower() + ') & (r6 = ' + str(state['r6']).lower() + \
                ') & (r7 = ' + str(state['r7']).lower() + ') & (r8 = ' + str(state['r8']).lower() + \
                ') & (r9 = ' + str(state['r9']).lower() + ') & (r10 = ' + str(state['r10']).lower() + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(ckbby\' = ' + str(next['ckbby']).lower() + ') & (r1\' = ' + str(next['r1']).lower() + \
                        ') & (r2\' = ' + str(next['r2']).lower() + \
                        ') & (r3\' = ' + str(next['r3']).lower() + ') & (r4\' = ' + str(next['r4']).lower() + \
                        ') & (r5\' = ' + str(next['r5']).lower() + ') & (r6\' = ' + str(next['r6']).lower() + \
                        ') & (r7\' = ' + str(next['r7']).lower() + ') & (r8\' = ' + str(next['r8']).lower() + \
                        ') & (r9\' = ' + str(next['r9']).lower() + ') & (r10\' = ' + str(next['r10']).lower() + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(ckbby\' = ' + str(state['ckbby']).lower() + ') & (r1\' = ' + str(state['r1']).lower() +  \
                    ') & (r2\' = ' + str(state['r2']).lower() + \
                    ') & (r3\' = ' + str(state['r3']).lower() + ') & (r4\' = ' + str(state['r4']).lower() + \
                    ') & (r5\' = ' + str(state['r5']).lower() + ') & (r6\' = ' + str(state['r6']).lower() + \
                    ') & (r7\' = ' + str(state['r7']).lower() + ') & (r8\' = ' + str(state['r8']).lower() + \
                    ') & (r9\' = ' + str(state['r9']).lower() + ') & (r10\' = ' + str(state['r10']).lower() + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if len(state['next']) > 0:
            data += '((ckbby = ' + str(state['ckbby']).lower() + ') & (r1 = ' + str(state['r1']).lower() + \
                ') & (r2 = ' + str(state['r2']).lower() + \
                ') & (r3 = ' + str(state['r3']).lower() + ') & (r4 = ' + str(state['r4']).lower() + \
                ') & (r5 = ' + str(state['r5']).lower() + ') & (r6 = ' + str(state['r6']).lower() + \
                ') & (r7 = ' + str(state['r7']).lower() + ') & (r8 = ' + str(state['r8']).lower() + \
                ') & (r9 = ' + str(state['r9']).lower() + ') & (r10 = ' + str(state['r10']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if len(state['next']) == 1 and state['next'][0]['is_dummy']:
            data += '((ckbby = ' + str(state['ckbby']).lower() + ') & (r1 = ' + str(state['r1']).lower() + \
                ') & (r2 = ' + str(state['r2']).lower() + \
                ') & (r3 = ' + str(state['r3']).lower() + ') & (r4 = ' + str(state['r4']).lower() + \
                ') & (r5 = ' + str(state['r5']).lower() + ') & (r6 = ' + str(state['r6']).lower() + \
                ') & (r7 = ' + str(state['r7']).lower() + ') & (r8 = ' + str(state['r8']).lower() + \
                ') & (r9 = ' + str(state['r9']).lower() + ') & (r10 = ' + str(state['r10']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(ckbby = ' + str(state['ckbby']).lower() + ') & (r1 = ' + str(state['r1']).lower() + \
                ') & (r2 = ' + str(state['r2']).lower() + \
                ') & (r3 = ' + str(state['r3']).lower() + ') & (r4 = ' + str(state['r4']).lower() + \
                ') & (r5 = ' + str(state['r5']).lower() + ') & (r6 = ' + str(state['r6']).lower() + \
                ') & (r7 = ' + str(state['r7']).lower() + ') & (r8 = ' + str(state['r8']).lower() + \
                ') & (r9 = ' + str(state['r9']).lower() + ') & (r10 = ' + str(state['r10']).lower() + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(ckbby = ' + str(state['ckbby']).lower() + ') & (r1 = ' + str(state['r1']).lower() + \
                ') & (r2 = ' + str(state['r2']).lower() + \
                ') & (r3 = ' + str(state['r3']).lower() + ') & (r4 = ' + str(state['r4']).lower() + \
                ') & (r5 = ' + str(state['r5']).lower() + ') & (r6 = ' + str(state['r6']).lower() + \
                ') & (r7 = ' + str(state['r7']).lower() + ') & (r8 = ' + str(state['r8']).lower() + \
                ') & (r9 = ' + str(state['r9']).lower() + ') & (r10 = ' + str(state['r10']).lower() + ');\n'
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
