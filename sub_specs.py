def sub_specs(variant):
    if variant == '0_dummy_data':
        return {'id':['000','001','002','003','004','005','006',], 'first_run':[[2,1],[2,1],[1,2],[2,1],[2,1],[2,1],[2,1]]}
    elif variant == 'one_gamble':
        return {'id':['000','001','002','003','004','005','006',], 'first_run':[[2,1],[2,1],[1,2],[2,1],[2,1],[2,1],[2,1]]}
    elif variant == 'two_gamble':
        return {'id':['000','001','002','003','004','005','006',], 'first_run':[[2,1],[2,1],[1,2],[2,1],[2,1],[2,1],[2,1]]}
    elif variant == 'two_gamble_new_c':
        return {'id':['000','001','002','003','004','005','006',], 'first_run':[[2,1],[2,1],[1,2],[2,1],[2,1],[2,1],[2,1]]}
    elif variant == 'two_gamble_hidden_wealth':
        return {'id':['000','001','002','003','004','005','006',], 'first_run':[[2,1],[2,1],[1,2],[2,1],[2,1],[2,1],[2,1]]}
    else:
        ValueError('Unknown variant')
