import argparse
from utils.utils import read_ground_truth, init
from env.maze_env import Maze
from evaluation import evaluate_episode_rtg
from evaluation.evaluate import Test
from model import DecisionTransformer, FeatureExtractor
import torch


def test(model, gt_dict,variant):

    device = variant['device']

    env = Maze()

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            with torch.no_grad():
                ret, length = evaluate_episode_rtg(
                    env,
                    act_dim=variant['act_dim'],
                    model=model,
                    scale=variant['scale'],
                    target_return=target_rew / variant['act_dim'],
                    device=device,
                    gt_dict=gt_dict,
                    variant=variant
                )
            returns.append(ret)
            lengths.append(length)
        return fn

    warmup_steps = variant['warmup_steps']

    optimizer = torch.optim.AdamW(model.parameters(), lr=variant['learning_rate'], weight_decay=variant['weight_decay'])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

    trainer = Test(
        model=model,
        scheduler=scheduler,
        eval_fns=[eval_episodes(variant['env_targets'])],
    )

    model.load_model(model, optimizer)
    # model.eval()

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)
        print(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='MH01', help="选择数据集['MH01','MH02'...]")

    parser.add_argument('--state_dim', type=int, default=64)
    parser.add_argument('--act_dim', type=int, default=6300)
    parser.add_argument('--max_ep_len', type=int, default=15000)
    parser.add_argument('--scale', type=int, default=1000.)
    parser.add_argument('--env_targets', type=int, default=3040)
    parser.add_argument('--warmup_steps', type=int, default=10)  # 预热步数，学习率会从`0`线性增加到初始设置的学习率
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=1)


    # 原始参数
    # parser.add_argument('--env', type=str, default='hopper')
    # parser.add_argument('--mode', type=str,default='normal')  # 标准设置为正常，稀疏设置为延迟
    # parser.add_argument('--K', type=int, default=20)
    # parser.add_argument('--pct_traj', type=float, default=1.)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--model_type', type=str, default='dt')  # dt for decision-transformer, bc for behavior cloning
    # parser.add_argument('--embed_dim', type=int, default=64)
    # parser.add_argument('--n_layer', type=int, default=3)
    # parser.add_argument('--n_head', type=int, default=1)
    # parser.add_argument('--activation_function', type=str, default='relu')
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    # parser.add_argument('--epoch', type=int, default=10, help="训练次数")

    args = parser.parse_args()

    init(args.dataset, args)

    feature_extractor = FeatureExtractor()

    model = DecisionTransformer(
        state_dim=args.state_dim,
        act_dim=args.act_dim,
        max_length=20,
        max_ep_len=args.max_ep_len,
        hidden_size=64,
        n_layer=3,
        n_head=1,
        n_inner=4 * 64,
        activation_function='relu',
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mem_size=20000,
        is_train=False,
        scale=1000,
        device='cuda'
    ).to(args.device)

    gt_dict = {}  # 初始化真值数据字典，避免重复加载
    read_ground_truth(gt_dict, args.gt_path)

    test(model, gt_dict, variant=vars(args))  # vars() 转换为dict
