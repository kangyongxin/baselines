source activate /data/kyx_data/GBMR/

CUDA_VISIBLE_DEVICES=0 python -m baselines.run --alg=deepq --env=CartPole-v0 --save_path=./cartpole_model.pkl --num_timesteps=1e5

CUDA_VISIBLE_DEVICES=1 python -m baselines.myrun --alg=valuebased --env=CartPole-v0 --save_path=./cartpole_model.pkl --num_timesteps=1e5

CUDA_VISIBLE_DEVICES=1 python -m baselines.run --alg=deepq --env=CartPole-v0 --save_path=/data/kyx_data/baselines/cartpole_model1.pkl  --num_timesteps=1e5

python -m baselines.run --alg=deepq --env=CartPole-v0 --load_path=/data/kyx_data/baselines/cartpole_model1.pkl --num_timesteps=0 --play

CUDA_VISIBLE_DEVICES=1 python -m baselines.myrun --alg=valuebased --env=CartPole-v0 --save_path=/data/kyx_data/baselines/cartpole_modelmy.pkl --num_timesteps=1e5
问题：
1.模型存在哪里了？
2.如何得到训练过程曲线？python baselines/visulize.py
3.如何看到训练之后的效果？--play

CUDA_VISIBLE_DEVICES=1 python -m baselines.myrun --alg=valuebased --env=PongNoFrameskip-v4 --save_path=/data/kyx_data/baselines/Pongmy.pkl --num_timesteps=1e5

CUDA_VISIBLE_DEVICES=1 python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --save_path=/data/kyx_data/baselines/Pongmy.pkl --num_timesteps=1e5

python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --load_path=/data/kyx_data/baselines/Pongmy.pkl --num_timesteps=0 --play