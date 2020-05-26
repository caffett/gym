# @Author: Zikang Xiong
# @Date:   2019-10-30 15:15:43
# @Last Modified by:   Zikang Xiong
# @Last Modified time: 2019-10-30 16:58:43
python AG.py -e Walker2DBulletEnv-v0 -A a2c -m 1000 -a 1000 -c True -v True
python AG.py -e Walker2DBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e Walker2DBulletEnv-v0 -A ddpg -m 1000 -a 1000 -c True -v True
python AG.py -e Walker2DBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True
python AG.py -e Walker2DBulletEnv-v0 -A td3 -m 1000 -a 1000 -c True -v True
python AG.py -e Walker2DBulletEnv-v0 -A trpo -m 1000 -a 1000 -c True -v True

python AG.py -e HalfCheetahBulletEnv-v0 -A a2c -m 1000 -a 1000 -c True -v True
python AG.py -e HalfCheetahBulletEnv-v0 -A acktr -m 1000 -a 1000 -c True -v True
python AG.py -e HalfCheetahBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e HalfCheetahBulletEnv-v0 -A ddpg -m 1000 -a 1000 -c True -v True
python AG.py -e HalfCheetahBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True
python AG.py -e HalfCheetahBulletEnv-v0 -A td3 -m 1000 -a 1000 -c True -v True
python AG.py -e HalfCheetahBulletEnv-v0 -A trpo -m 1000 -a 1000 -c True -v True

python AG.py -e AntBulletEnv-v0 -A a2c -m 1000 -a 1000 -c True -v True
python AG.py -e AntBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e AntBulletEnv-v0 -A ddpg -m 1000 -a 1000 -c True -v True
python AG.py -e AntBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True
python AG.py -e AntBulletEnv-v0 -A td3 -m 1000 -a 1000 -c True -v True
python AG.py -e AntBulletEnv-v0 -A trpo -m 1000 -a 1000 -c True -v True

python AG.py -e InvertedDoublePendulumBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e InvertedDoublePendulumBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True
python AG.py -e InvertedDoublePendulumBulletEnv-v0 -A td3 -m 1000 -a 1000 -c True -v True

python AG.py -e InvertedPendulumSwingupBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e InvertedPendulumSwingupBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True
python AG.py -e InvertedPendulumSwingupBulletEnv-v0 -A td3 -m 1000 -a 1000 -c True -v True

python AG.py -e ReacherBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e ReacherBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True

python AG.py -e HopperBulletEnv-v0 -A a2c -m 1000 -a 1000 -c True -v True
python AG.py -e HopperBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e HopperBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True
python AG.py -e HopperBulletEnv-v0 -A td3 -m 1000 -a 1000 -c True -v True
python AG.py -e HopperBulletEnv-v0 -A trpo -m 1000 -a 1000 -c True -v True

python AG.py -e HumanoidBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True
python AG.py -e HumanoidBulletEnv-v0 -A sac -m 1000 -a 1000 -c True -v True
python AG.py -e HumanoidBulletEnv-v0 -A td3 -m 1000 -a 1000 -c True -v True

# python AG.py -e MinitaurBulletEnv-v0 -A ppo2 -m 1000 -a 1000 -c True -v True


