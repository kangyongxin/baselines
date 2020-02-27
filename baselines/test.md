

1 以DQN为例，梳理baseline 的使用流程
2区分 # DQN VS. Double DQN的不同之处

1 训练
python -m baselines.run --alg=deepq --env=CartPole-v0 --save_path=./cartpole_model.pkl --num_timesteps=1e5
使用
python -m baselines.run --alg=deepq --env=CartPole-v0 --load_path=./cartpole_model.pkl --num_timesteps=0 --play

训练入口 baseline.run 
+ 系统包加载
+ baseline.common 中的一些包的加载
+ main()
     + 读入参数
     + model, env = train(args, extra_args)
     + model.save()
     + play
+ train()
    + 读取环境
    + learn = get_learn_function(args.alg) //deepq
    + env = build_env(args)
    + alg_kwargs['network'] = args.network //没有就默认
    + model = learn()
+ learn = get_learn_function()
    + get_alg_module(alg).learn
        +  first try to import the alg module from baselines ：alg_module = import_module('.'.join(['baselines', alg, submodule]))//如果想写自己的函数在baseline的下面再建立一个文件夹，文件中要learn 函数即可。
+ build_env()
    + env_type, env_id = get_env_type(args) //读入基本信息
    + 如果是deepq or trpo_mpi要分别制作环境，否则都是统一的4帧一次堆叠
+ model = learn()
    + 以deepa中的learn 为例
    + q_func = build_q_func(network, **network_kwargs)
    + deepq.build_train()
    + Create the replay buffer;是否 prioritized 区别
    + U.initialize()
    + update_target()
    + 询问是否存储模型
    + for t in range(total_timesteps):
        + 先讨论是否有模型噪声
        + action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0] //执行动作，act是个网络吗
        + new_obs, rew, done, _ = env.step(env_action)
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        + episode_rewards[-1] += rew
        + if t > learning_starts and t % train_freq == 0:
            + 训练不是每次都有的，而是要隔一段时间按进行一次
            + 每次通过不同的方式得到一个bacth 的数据
            + td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
        + if t > learning_starts and t % target_network_update_freq == 0:
            + 更新也不是每次都有的
            + update_target()
+ q_func = build_q_func()
    + 在models中定义，返回一个q_out
    + 这里定义了这一通操作是干啥的,定义网络结构，从状态动作，到Q值的映射方法。
    + dueling 定义在这里，默认是true，不过目前想关注的是double
+ deepq.build_train()
    + 在build_graph 内的， double_q默认True
    + act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse) //这几个输入啥意思
        + 根据输入和随机 还有当前的q值得到当前动作
    +  q network evaluation ： q_t = q_func() //上一个函数构建的函数，在这里用
    + target q network evalution ：q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")//这两个东西的结构是相同的
    + q_t_selected//已经选择的动作的Q值
    
    + compute estimate of best possible value starting from state at t + 1：
    + 在这里区分是否使用double q, 输出一个q_tp1_best;
    + double q中，输入是当前网络最大的动作值所对应的动作，在评估网络中对应的Q值
    + dqn中是评估网络中最大的值
    + 然后compute RHS of bellman equation ：q_t_selected_target
    + 算TDerror, 并加入权重
    + train = U.function产生可以调用的函数，q_values = U.function([obs_t_input], q_t)
    + return act_f, train, update_target, {'q_values': q_values}





