# @Author: Zikang Xiong
# @Date:   2019-10-30 17:11:45
# @Last Modified by:   Zikang Xiong
# @Last Modified time: 2019-11-11 15:40:22

python AG.py -e Pendulum-v0 -A a2c -i 100 -m 200 -a 200 -c True -v False

python AG.py -e CartPole-v1 -A a2c -i 1000 -m 500 -a 500 -c True -v False
python AG.py -e CartPole-v1 -A acktr -i 1000 -m 500 -a 500 -c True -v False
python AG.py -e CartPole-v1 -A ppo2 -i 1000 -m 500 -a 500 -c True -v False
python AG.py -e CartPole-v1 -A trpo -i 1000 -m 500 -a 500 -c True -v False
python AG.py -e CartPole-v1 -A acer -i 1000 -m 500 -a 500 -c True -v False
python AG.py -e CartPole-v1 -A dqn -i 1000 -m 500 -a 500 -c True -v False
