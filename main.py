import argparse
import os

from Utils.agent_utils import MODEL_MAP, get_agent
from Utils.utils import init,read_ground_truth, get_true_traj
from Utils.evaluate_episodes import set_params
from maze_env import Maze
from Utils.compute_rmse import compute_rmse_with_file
# from Utils.evaluate_episodes import write_my_params

#------------初始化参数设置
def get_args():
    '''
    参数定义
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='corridor4', help="选择数据集['MH01','MH02'...]")
    parser.add_argument("--model", type=str, default="DTQN", choices=list(MODEL_MAP.keys()), help="创建DTQN模型")
    parser.add_argument("--envs", type=str, default="Maze", help="调用Maze()创建环境")
    parser.add_argument("--obs_embed", type=int, default=4096, help="状态输入是tensor")
    parser.add_argument("--a_embed", type=int, default=1, help="动作值的输入,默认为1.2,0.8,0.9的Index，范围是0-6300，是int类型的1维度")
    parser.add_argument("--in_embed", type=int, default=4100, help="The dimensionality of the network. In the transformer, this is referred to as `d_model`.")
    # parser.add_argument("--buf_size", type=int, default=16, help="Number of timesteps to store in replay buffer.")
    parser.add_argument("--buf_size", type=int, default=64, help="Number of timesteps to store in replay buffer.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--context", type=int, default=8, help="For DTQN, the context length to use to train the network.")
    parser.add_argument("--max_episode_steps", type=int, default=8, help="The maximum number of steps allowed in the environment.")
    parser.add_argument("--history", type=int, default=6, help="This is how many (intermediate) Q-values we use to train for each context. To turn off intermediate Q-value prediction, set `--history 1`. To use the entire context, set history equal to the context length.")
    parser.add_argument("--tuf", type=int, default=50, help="How many steps between each (hard) target network update.每50步更新一次")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
    # DTQN 的参数定义
    parser.add_argument("--heads", type=int, default=4, help="Number of heads to use for the transformer.")
    parser.add_argument("--layers", type=int, default=2, help="Number of transformer blocks to use for the transformer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")
    parser.add_argument("--identity", action="store_true", help="Whether or not to use identity map reordering.")
    parser.add_argument("--gate", type=str, default="res", choices=["res", "gru"], help="Combine step to use.")
    parser.add_argument("--pos", default="learned", choices=["learned", "sin", "none"], help="The type of positional encodings to use.")
    parser.add_argument("--bag_size", type=int, default=0, help="The size of the persistent memory bag.")

    args = parser.parse_args()  # 获取解析后的参数
    if args.dataset == "V102":
        epo = 1710 + 50
    elif args.dataset == "MH01":
        epo = 3685
    elif args.dataset == "MH02":
        epo = 3040
    elif args.dataset == "MH03":
        epo = 2750
    elif args.dataset == "MH04":
        epo = 2033+50
    elif args.dataset == "MH05":
        epo = 2273+50
    elif args.dataset == "V101":
        epo = 2912+50
    elif args.dataset == "V103":
        epo = 2150+50
    elif args.dataset == "V201":
        epo = 2280+50
    elif args.dataset == "V202":
        epo = 2348+50
    elif args.dataset == "V203":
        epo = 1922+50
    elif args.dataset == "corridor1":
        epo = 6000+100
    elif args.dataset == "corridor2":
        epo = 6800+100
    elif args.dataset == "corridor3":
        epo = 5900+100
    elif args.dataset == "corridor4":
        epo = 2000+100
    elif args.dataset == "corridor5":
        epo = 6000+100
    elif args.dataset == "room1":
        epo = 3000+100
    elif args.dataset == "room2":
        epo = 3000+100
    elif args.dataset == "room3":
        epo = 3000+100
    elif args.dataset == "room4":
        epo = 2500+100
    elif args.dataset == "room5":
        epo = 3000+100
    elif args.dataset == "room6":
        epo = 2900



    return parser.parse_args(), epo






prev_ob_name = None
# -----------step
def step(agent, env: Maze, eps, args, gt_dict, numparams, numparams_eps) -> bool:
    """Use the agent's policy to get the next action, take it, and then record the result.

    Arguments:
        agent:  the agent to use.
        env:    gym.Env
        eps:    the epsilon value (for epsilon-greedy policy)

    Returns:
        done: bool, whether or not the episode has finished.
    """
    # print(numparams, numparams_eps)
    global prev_ob_name
    action = agent.get_action(epsilon=eps)
    # print(action)
    current_bag_v = 0
    next_ob_name, next_ob, reward, succ = env.step(action, args, gt_dict)
    current_bag_v += 1
    action_next = env.parameter_space[action]


    if prev_ob_name != next_ob_name:
        set_params(numparams, numparams_eps, action_next, args)
        numparams += 1
        numparams_eps += 1
    done = False
    if current_bag_v < args.bag_size:
        done = True
    # agent.observe(next_ob, action, reward, done)

    # agent.observe(next_obs, action, reward, buffer_done)
    # print(reward)
    next_action = action
    return next_ob_name, next_ob, next_action ,reward, succ, numparams, numparams_eps

#----------主函数--------
def experiment():
    args,epo = get_args()

    datasetname = args.dataset
    init(datasetname, args)
    gt_dict = {}
    # --------- 获取真实轨迹gt_dict
    read_ground_truth(gt_dict, args.gt_path)

    env = Maze()

    # state_dim = env.n_features
    # act_dim = env.n_actions
    #--------- 获取真实轨迹true_traj
    global true_traj
    global prev_ob_name
    true_traj = get_true_traj(args.gt_path)
    #--------- 测试状态获取下一帧的时间戳，图片具体内容，奖励
    # next_frame_name, ob, reward, succ = get_state_and_reward(gt_dict, args)


    # getres_img(gt_dict,args)


    # print(next_frame_name, ob, reward)
# --------------- 以下代码实现了测试和ORB SLAM代码交互，交互的原理就是写入参数的前两列的数字不断增大--------------------------------
#     numparams = 0
#     numparams_eps = 1
# #
#     params = [1.2, 8, 0.9]
# #
#     succ = True
#     while succ == True:
#         set_params(numparams, numparams_eps, params, args)
#         next_frame_name, ob, reward = get_state_and_reward(gt_dict, args)
#         numparams += 1
#         numparams_eps += 1
#

# 以下内容需要解开注释
# ------------------ 接下来就是构建智能体
    agent = get_agent(args.model,
                      args.envs,
                      args.obs_embed,
                      args.a_embed,
                      args.in_embed,
                      args.buf_size,
                      args.device,
                      args.lr,
                      args.batch,
                      args.context,
                      args.max_episode_steps,
                      args.history,
                      args.tuf,
                      args.discount,
                      # DTQN
                      args.heads,
                      args.layers,
                      args.dropout,
                      args.identity,
                      args.gate,
                      args.pos,
                      args.bag_size
                      )
    print("智能体获取成功")

#-----------------训练模块---------------
    agent.eval_off()
    agent.context_reset(env.reset())
    # agent.eval_on()
    succ = True
    eps = 0.0
    t = 1
    done = False
    numparams = 0
    numparams_eps = 1
    succ = True
    while succ == True and t < epo:
        succ = False
        next_frame_name, next_frame_name_obs, next_action, reward, succ, numparams, numparams_eps = step(agent, env, eps, args, gt_dict, numparams, numparams_eps)
        agent.observe(next_frame_name_obs, next_action, reward, done)
        for i in range(4):
            agent.train()
        t = t+1


        # if(t == 200):# 如果达到200步就停止训练并保存模型
        #     agent.save_model('/home/mxb/DTQN/mydtqn/DTQN_SLAM/checkpoint/', args)
        #     break
        # params = env.parameter_space[action
        # set_params(numparams, numparams_eps, params, args)
        if t%(epo-500) == 0:
            agent.save_model('/home/wwp/mxb/DTQN_SLAM/model/tum/', args)

    # 保存模型参数
    # torch.save(agent, args.dataset + '_model.pth')
    # agent.save_model('/home/mxb/DTQN/mydtqn/DTQN_SLAM/checkpoint/', args)

def test(args,epo):

    datasetname = args.dataset
    init(datasetname, args)
    gt_dict = {}
    print(args.dataset)
    agent = get_agent(args.model,
                      args.envs,
                      args.obs_embed,
                      args.a_embed,
                      args.in_embed,
                      args.buf_size,
                      args.device,
                      args.lr,
                      args.batch,
                      args.context,
                      args.max_episode_steps,
                      args.history,
                      args.tuf,
                      args.discount,
                      # DTQN
                      args.heads,
                      args.layers,
                      args.dropout,
                      args.identity,
                      args.gate,
                      args.pos,
                      args.bag_size
                      )
    # 读取地面真实数据
    read_ground_truth(gt_dict, args.gt_path)
    agent.eval_on()
    env = Maze()
    agent.context_reset(env.reset())
    # 加载模型
    #state_dict = torch.load('/home/mxb/DTQN/mydtqn/DTQN_SLAM/checkpoint/' + args.dataset + '.pt')
    #agent.load_state_dict(state_dict)
    #agent.load_model('/home/mxb/DTQN/mydtqn/DTQN_SLAM/checkpoint/', args)
    # agent.eval_on()

    # 进行测试
    succ = True
    eps = 0.0
    t = 1
    done = False
    numparams = 0
    numparams_eps = 1
    params = [1.34, 3, 0.77]
    # set_params(numparams, numparams_eps, params, args)
    agent.load_model('/home/wwp/mxb/DTQN_SLAM/model/tum/', args)
    while succ == True:
        next_frame_name, next_frame_name_obs, next_action, reward, succ, numparams, numparams_eps = step(agent,
                                                                                                                env,
                                                                                                                eps,
                                                                                                                args,
                                                                                                                gt_dict,
                                                                                                                numparams,
                                                                                                                numparams_eps)
        # if numparams_eps % 50 == 0:
        agent.observe(next_frame_name_obs, next_action, reward, done)
        # 给wwp写参数使用
        # action_next = env.parameter_space[next_action]
        # write_my_params(next_frame_name, action_next[0], action_next[1], action_next[2])
        # params = [1.2, 8, 0.9]
        # set_params(numparams, numparams_eps, params, args)
        # numparams=numparams+1
        # numparams_eps=numparams_eps+1
        # agent.train()
        t = t + 1
if __name__ == '__main__':

    args, epo = get_args()
    #experiment()
    test(args,epo)





