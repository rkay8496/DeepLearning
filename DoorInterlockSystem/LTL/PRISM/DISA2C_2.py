import copy
import os
import json


def read_file():
    f = open(os.getcwd() + '/../DoorInterlockSystemA2C_2_traces.json', 'r')
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
            'closed': False,
            'open': False,
            'partially': False,
            'power': False,
            'off': False,
            'on': False,
            'is_dummy': True,
        }
        item = {}
        for i in range(len(obj['closed'])):
            item = {
                'No': -1,
                'closed': obj['closed'][i][1],
                'open': obj['open'][i][1],
                'partially': obj['partially'][i][1],
                'power': obj['power'][i][1],
                'off': obj['off'][i][1],
                'on': obj['on'][i][1],
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
                    elif l['closed'] == j['closed'] and l['partially'] == j['partially'] and \
                            l['open'] == j['open'] and l['power'] == j['power'] and \
                            l['off'] == j['off'] and l['on'] == j['on']:
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
    f = open('./DISA2C_2_LTL.prism', 'w')
    data = 'dtmc\n' \
            '\n' \
            'module DIS\n' \
            '\t\n' \
            '\tclosed : bool init true;\n' \
            '\tpartially : bool init false;\n' \
            '\topen : bool init false;\n' \
            '\tpower : bool init true;\n' \
            '\toff : bool init false;\n' \
            '\ton : bool init true;\n' \
            '\n'
    for state in state_list:
        data += '\t[] '
        data += '(closed = ' + str(state['closed']).lower() + ') & (partially = ' + str(state['partially']).lower() + \
                ') & (open = ' + str(state['open']).lower() + ') & (power = ' + str(state['power']).lower() + \
                ') & (off = ' + str(state['off']).lower() + ') & (on = ' + str(state['on']).lower() + ') -> '
        if len(state['next']) > 0:
            for next in state['next']:
                data += str(next['probability']) + ':(closed\' = ' + str(next['closed']).lower() + ') & (partially\' = ' + str(next['partially']).lower() + \
                        ') & (open\' = ' + str(next['open']).lower() + ') & (power\' = ' + str(next['power']).lower() + \
                        ') & (off\' = ' + str(next['off']).lower() + ') & (on\' = ' + str(next['on']).lower() + ') + '
            data = data[:-3]
            data += ';\n'
        else:
            data += '1:(closed\' = ' + str(state['closed']).lower() + ') & (partially\' = ' + str(state['partially']).lower() + \
                    ') & (open\' = ' + str(state['open']).lower() + ') & (power\' = ' + str(state['power']).lower() + \
                    ') & (off\' = ' + str(state['off']).lower() + ') & (on\' = ' + str(state['on']).lower() + ')'
            data += ';\n'
    data += 'endmodule\n'
    data += '\n'
    data += 'label \"safe\" = '
    for state in state_list:
        if len(state['next']) > 0:
            data += '((closed = ' + str(state['closed']).lower() + ') & (partially = ' + str(state['partially']).lower() + \
                    ') & (open = ' + str(state['open']).lower() + ') & (power = ' + str(state['power']).lower() + \
                    ') & (off = ' + str(state['off']).lower() + ') & (on = ' + str(state['on']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    data += 'label \"fail\" = '
    for state in state_list:
        if len(state['next']) == 1 and state['next'][0]['is_dummy']:
            data += '((closed = ' + str(state['closed']).lower() + ') & (partially = ' + str(state['partially']).lower() + \
                    ') & (open = ' + str(state['open']).lower() + ') & (power = ' + str(state['power']).lower() + \
                    ') & (off = ' + str(state['off']).lower() + ') & (on = ' + str(state['on']).lower() + ')) | '
    data = data[:-3]
    data += ';\n'
    for state in state_list:
        if not state['is_dummy']:
            data += 'label \"s' + str(state['No']) + '\" = ' + '(closed = ' + str(state['closed']).lower() + ') & (partially = ' + str(state['partially']).lower() + \
                    ') & (open = ' + str(state['open']).lower() + ') & (power = ' + str(state['power']).lower() + \
                    ') & (off = ' + str(state['off']).lower() + ') & (on = ' + str(state['on']).lower() + ');\n'
    for state in state_list:
        if state['is_dummy']:
            data += 'label \"dummy' + str(state['No']) + '\" = ' + '(closed = ' + str(state['closed']).lower() + ') & (partially = ' + str(state['partially']).lower() + \
                    ') & (open = ' + str(state['open']).lower() + ') & (power = ' + str(state['power']).lower() + \
                    ') & (off = ' + str(state['off']).lower() + ') & (on = ' + str(state['on']).lower() + ');\n'
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
