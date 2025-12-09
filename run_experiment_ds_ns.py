from experiment import execute_experiment_mnist4, execute_experiment_cifar, \
    execute_experiment_acasxu, execute_experiment_mnistConv
import sys

# delta_eps = 1/256 * d_eps
# eps = 1/256 * d_eps * i_eps


def run_exp(dataset, d_eps, i_eps, net_idx1=None, net_idx2=None, DS_mode=None, NS_mode=None, time=None, exe_start=0, exe_end=10, inputs_num=10):
    if dataset == "acasxu":
        time = 420
        threshold_analysis = True
        execute_experiment_acasxu(net_idx1=net_idx1, net_idx2=net_idx2, d_eps=d_eps, DS_mode=DS_mode, NS_mode=NS_mode,
                                  split_limit=100, exe_start=exe_start, exe_end=exe_end, time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "mnist4":
        if time is None:
            time = 600
        threshold_analysis = True
        execute_experiment_mnist4(d_eps=d_eps, i_eps=i_eps, DS_mode=DS_mode, NS_mode=NS_mode, split_limit=100, exe_start=exe_start,
                                  exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "mnistConv":
        if time is None:
            time = 600
        threshold_analysis = True
        execute_experiment_mnistConv(d_eps=d_eps, i_eps=i_eps, DS_mode=DS_mode, NS_mode=NS_mode, split_limit=100, exe_start=exe_start,
                                     exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis)
    elif dataset == "cifar":
        threshold_analysis = True
        execute_experiment_cifar(d_eps=d_eps, i_eps=i_eps, DS_mode=DS_mode, NS_mode=NS_mode, split_limit=100, exe_start=exe_start,
                                 exe_end=exe_end, inputs_num=inputs_num, time_budget=time, threshold_analysis=threshold_analysis)
    else:
        print("Invalid dataset name. Use 'mnist2', 'mnist4', 'mnistConv', 'cifar', or 'acasxu'.")
        sys.exit(1)


for d_val in [10]:
    for net_idx1 in [1, 2]:
        for net_idx2 in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            for dsns_mode in ['DS_random_Z', 'DS_dual_Z', 'NS_dual', 'NS_dual_ind']:
                print(f"Running experiments with d_eps={d_val}, net_idx1={net_idx1}, net_idx2={net_idx2}, DS_mode={dsns_mode}")
                # acasxu
                print("** Run acasxu **")
                if dsns_mode.startswith('DS'):
                    run_exp("acasxu", DS_mode=dsns_mode, d_eps=d_val, i_eps=None, net_idx1=net_idx1, net_idx2=net_idx2)
                elif dsns_mode.startswith('NS'):
                    run_exp("acasxu", NS_mode=dsns_mode, d_eps=d_val, i_eps=None, net_idx1=net_idx1, net_idx2=net_idx2)


for d_val in [1, 2, 3]:
    for i_val in [2, 3, 4]:
        for dsns_mode in ['DS_random_Z', 'DS_dual_Z', 'NS_dual', 'NS_dual_ind']:
            print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, DS/NS mode={dsns_mode}")

            # mnist4
            print("** Run mnist4 **")
            exe_start = 0
            exe_end = 13  # run [0, 1, ..., 12]
            inputs_num = 13
            if dsns_mode.startswith('DS'):
                run_exp("mnist4", DS_mode=dsns_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)  # run [0, 1, ..., 12]
            elif dsns_mode.startswith('NS'):
                run_exp("mnist4", NS_mode=dsns_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)  # run [0, 1, ..., 12]

            # mnistConv
            print("** Run mnistConv **")
            exe_start = 0
            exe_end = 13  # run [0, 1, ..., 12]
            inputs_num = 13
            if dsns_mode.startswith('DS'):
                run_exp("mnistConv", DS_mode=dsns_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)  # run [0, 1, ..., 12]
            elif dsns_mode.startswith('NS'):
                run_exp("mnistConv", NS_mode=dsns_mode, d_eps=d_val, i_eps=i_val, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)  # run [0, 1, ..., 12]


for d_val in [1, 2]:
    for i_val in [2, 3, 4]:
        for dsns_mode in ['DS_random_Z', 'DS_dual_Z', 'NS_dual', 'NS_dual_ind']:
            print(f"Running experiments with d_eps={d_val}, i_eps={i_val}, DS/NS mode={dsns_mode}")

            # cifar
            print("** Run cifar **")
            if d_val == 1:
                time = 1800
            elif d_val == 2:
                time = 3600
            else:
                raise ValueError("d_val should be 1, 2, or 3")
            d_eps = d_val
            i_eps = i_val
            exe_start = 0
            exe_end = 16  # run [0, 1, ..., 15]
            inputs_num = 16
            if dsns_mode.startswith('DS'):
                run_exp("cifar", DS_mode=dsns_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)
            elif dsns_mode.startswith('NS'):
                run_exp("cifar", NS_mode=dsns_mode, d_eps=d_val, i_eps=i_val, time=time, exe_start=exe_start, exe_end=exe_end, inputs_num=inputs_num)

print("Done!")
